//! Antigravity CLI provider that routes every prompt through `agy acp`.
//!
//! Jcode acts as an Agent Client Protocol (ACP) *client*: it spawns Google's
//! official Antigravity CLI (`agy acp`) as a stdio JSON-RPC server and sends
//! user turns with `session/prompt`. Jcode never calls Google's model APIs
//! directly and never touches the CLI's OAuth tokens, so authentication,
//! quota, tool execution, and the system prompt all stay inside the official
//! client. Jcode only renders the streamed `session/update` notifications.
//!
//! Because the CLI runs its own tools, this provider reports
//! `handles_tools_internally() == true` and surfaces CLI tool calls as
//! already-executed tool events.

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use jcode_message_types::{ContentBlock, Message, Role, StreamEvent, ToolDefinition};
use jcode_provider_core::{EventStream, Provider};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex, RwLock};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

pub const PROVIDER_NAME: &str = "antigravity-cli";
pub const DEFAULT_MODEL: &str = "default";
const ACP_PROTOCOL_VERSION: u64 = 1;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Offline fallback only. The live catalog comes from `session/new`.
pub const FALLBACK_MODELS: &[&str] = &[DEFAULT_MODEL];

static DISCOVERED_MODELS: LazyLock<RwLock<Vec<(String, String)>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));

/// Binary to launch. Override with `JCODE_ANTIGRAVITY_CLI` (e.g. a full path).
fn cli_binary() -> String {
    std::env::var("JCODE_ANTIGRAVITY_CLI")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "agy".to_string())
}

/// ACP session mode. Defaults to `auto` so the CLI can work unattended. Set
/// `JCODE_ANTIGRAVITY_MODE` to `default`, `accept-edits`, `auto`, or `plan`.
fn session_mode() -> String {
    std::env::var("JCODE_ANTIGRAVITY_MODE")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "auto".to_string())
}

pub fn cli_available() -> bool {
    let bin = cli_binary();
    if bin.contains('/') {
        return std::path::Path::new(&bin).exists();
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(&bin).is_file()))
        .unwrap_or(false)
}

type Pending = Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>;
type UpdateSink = Arc<StdMutex<Option<mpsc::UnboundedSender<Value>>>>;

/// One running `agy acp` process.
struct AcpConnection {
    _child: StdMutex<Child>,
    stdin: Mutex<ChildStdin>,
    next_id: AtomicU64,
    pending: Pending,
    updates: UpdateSink,
}

impl AcpConnection {
    async fn spawn(cwd: &std::path::Path) -> Result<Arc<Self>> {
        let bin = cli_binary();
        let mut child = Command::new(&bin)
            .arg("acp")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!("failed to launch `{bin} acp`. Install the Antigravity CLI or set JCODE_ANTIGRAVITY_CLI")
            })?;
        let stdin = child.stdin.take().context("agy acp stdin unavailable")?;
        let stdout = child.stdout.take().context("agy acp stdout unavailable")?;
        let conn = Arc::new(Self {
            _child: StdMutex::new(child),
            stdin: Mutex::new(stdin),
            next_id: AtomicU64::new(1),
            pending: Arc::new(StdMutex::new(HashMap::new())),
            updates: Arc::new(StdMutex::new(None)),
        });
        let weak = Arc::downgrade(&conn);
        let pending = conn.pending.clone();
        let updates = conn.updates.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                let has_method = msg.get("method").and_then(Value::as_str).is_some();
                let id = msg.get("id").cloned();
                if has_method && id.is_some() {
                    // Agent -> client request (permissions, fs, terminal).
                    if let Some(conn) = weak.upgrade() {
                        let reply = handle_agent_request(&msg);
                        let _ = conn.write(&reply).await;
                    }
                } else if has_method {
                    if msg["method"] == "session/update"
                        && let Some(tx) = updates.lock().unwrap().as_ref()
                    {
                        let _ = tx.send(msg["params"].clone());
                    }
                } else if let Some(id) = id.and_then(|v| v.as_u64()) {
                    if let Some(tx) = pending.lock().unwrap().remove(&id) {
                        let result = if let Some(err) = msg.get("error") {
                            Err(anyhow!("Antigravity ACP error: {err}"))
                        } else {
                            Ok(msg.get("result").cloned().unwrap_or(Value::Null))
                        };
                        let _ = tx.send(result);
                    }
                }
            }
            // Process exited: fail everything still waiting.
            for (_, tx) in pending.lock().unwrap().drain() {
                let _ = tx.send(Err(anyhow!("Antigravity CLI (agy acp) exited")));
            }
            updates.lock().unwrap().take();
        });
        Ok(conn)
    }

    async fn write(&self, msg: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(msg)?;
        line.push(b'\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(&line).await?;
        stdin.flush().await?;
        Ok(())
    }

    fn start_request(&self, method: &str, params: Value) -> (u64, oneshot::Receiver<Result<Value>>, Value) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        (id, rx, msg)
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let (id, rx, msg) = self.start_request(method, params);
        self.write(&msg).await?;
        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow!("Antigravity CLI connection closed")),
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                Err(anyhow!("Antigravity ACP `{method}` timed out"))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.write(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    async fn initialize(&self) -> Result<Value> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": {"readTextFile": false, "writeTextFile": false},
                    "terminal": false
                },
                "clientInfo": {"name": "jcode", "version": env!("CARGO_PKG_VERSION")}
            }),
        )
        .await
    }
}

/// Answer agent-initiated requests. Permissions are approved (preferring a
/// one-time allow) because jcode itself runs tools without confirmation.
/// We advertise no fs/terminal capability, so anything else is refused.
fn handle_agent_request(msg: &Value) -> Value {
    let id = msg["id"].clone();
    if msg["method"] == "session/request_permission" {
        let options = msg["params"]["options"].as_array().cloned().unwrap_or_default();
        let pick = ["allow_once", "allow_always"].iter().find_map(|kind| {
            options
                .iter()
                .find(|o| o["kind"] == *kind)
                .and_then(|o| o["optionId"].as_str())
        });
        let outcome = match pick {
            Some(option_id) => json!({"outcome": "selected", "optionId": option_id}),
            None => json!({"outcome": "cancelled"}),
        };
        return json!({"jsonrpc": "2.0", "id": id, "result": {"outcome": outcome}});
    }
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "method not supported by jcode"}})
}

fn parse_models(session_new: &Value) -> Vec<(String, String)> {
    session_new["models"]["availableModels"]
        .as_array()
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m["modelId"].as_str()?.to_string();
                    let name = m["name"].as_str().unwrap_or(&id).to_string();
                    Some((id, name))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn store_models(models: Vec<(String, String)>) {
    if !models.is_empty() {
        *DISCOVERED_MODELS.write().unwrap() = models;
    }
}

fn discovered_model_ids() -> Vec<String> {
    DISCOVERED_MODELS
        .read()
        .unwrap()
        .iter()
        .map(|(id, _)| id.clone())
        .collect()
}

struct AcpSession {
    conn: Arc<AcpConnection>,
    session_id: String,
    applied_model: Option<String>,
}

pub struct AntigravityCliProvider {
    model: Arc<RwLock<String>>,
    session: Arc<Mutex<Option<AcpSession>>>,
}

impl Default for AntigravityCliProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AntigravityCliProvider {
    pub fn new() -> Self {
        let model = std::env::var("JCODE_ANTIGRAVITY_CLI_MODEL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        Self::with_model(&model)
    }

    pub fn with_model(model: &str) -> Self {
        Self {
            model: Arc::new(RwLock::new(model.to_string())),
            session: Arc::new(Mutex::new(None)),
        }
    }

    async fn open_session(conn: Arc<AcpConnection>, cwd: &std::path::Path) -> Result<AcpSession> {
        conn.initialize().await?;
        let created = conn
            .request(
                "session/new",
                json!({"cwd": cwd.to_string_lossy(), "mcpServers": []}),
            )
            .await
            .context("Antigravity CLI session/new failed. Run `agy` once to sign in with Google")?;
        store_models(parse_models(&created));
        let session_id = created["sessionId"]
            .as_str()
            .context("Antigravity session/new returned no sessionId")?
            .to_string();
        let mode = session_mode();
        let _ = conn
            .request(
                "session/set_mode",
                json!({"sessionId": session_id, "modeId": mode}),
            )
            .await;
        Ok(AcpSession {
            conn,
            session_id,
            applied_model: None,
        })
    }

    async fn ensure_model(session: &mut AcpSession, model: &str) -> Result<()> {
        if model == DEFAULT_MODEL || session.applied_model.as_deref() == Some(model) {
            return Ok(());
        }
        let sid = session.session_id.clone();
        let result = session
            .conn
            .request("session/set_model", json!({"sessionId": sid, "modelId": model}))
            .await;
        if result.is_err() {
            session
                .conn
                .request(
                    "session/set_config_option",
                    json!({"sessionId": sid, "configId": "model", "value": model}),
                )
                .await
                .with_context(|| format!("Antigravity CLI rejected model `{model}`"))?;
        }
        session.applied_model = Some(model.to_string());
        Ok(())
    }
}

fn render_block(out: &mut String, block: &ContentBlock) {
    match block {
        ContentBlock::Text { text, .. } => {
            out.push_str(text);
            out.push('\n');
        }
        ContentBlock::ToolResult { content, .. } => {
            out.push_str("[tool result]\n");
            out.push_str(content);
            out.push('\n');
        }
        _ => {}
    }
}

/// Text to send for this turn. On a fresh ACP session with prior history
/// (resumed jcode session, provider switch) the transcript is replayed as
/// context. Otherwise only the trailing non-assistant messages are sent,
/// since the CLI session already holds everything earlier.
fn build_turn_prompt(messages: &[Message], fresh_session: bool) -> String {
    let tail_start = messages
        .iter()
        .rposition(|m| m.role == Role::Assistant)
        .map(|i| i + 1)
        .unwrap_or(0);
    let mut out = String::new();
    if fresh_session && tail_start > 0 {
        out.push_str("<previous_conversation>\n");
        for message in &messages[..tail_start] {
            out.push_str(match message.role {
                Role::User => "User:\n",
                Role::Assistant => "Assistant:\n",
            });
            for block in &message.content {
                render_block(&mut out, block);
            }
            out.push('\n');
        }
        out.push_str("</previous_conversation>\n\n");
    }
    for message in &messages[tail_start..] {
        for block in &message.content {
            render_block(&mut out, block);
        }
    }
    out.trim_end().to_string()
}

fn content_text(content: &Value) -> Option<String> {
    match content["type"].as_str()? {
        "text" => content["text"].as_str().map(str::to_string),
        _ => None,
    }
}

fn tool_output_text(update: &Value) -> String {
    let mut out = String::new();
    if let Some(items) = update["content"].as_array() {
        for item in items {
            match item["type"].as_str() {
                Some("content") => {
                    if let Some(t) = content_text(&item["content"]) {
                        out.push_str(&t);
                    }
                }
                Some("diff") => {
                    out.push_str(&format!(
                        "edited {}\n",
                        item["path"].as_str().unwrap_or("file")
                    ));
                }
                _ => {}
            }
        }
    }
    if out.is_empty()
        && let Some(raw) = update.get("rawOutput").filter(|v| !v.is_null())
    {
        out = match raw.as_str() {
            Some(s) => s.to_string(),
            None => raw.to_string(),
        };
    }
    out
}

/// Translate one `session/update` into jcode stream events.
fn translate_update(
    update: &Value,
    thinking: &mut bool,
    open_tools: &mut HashMap<String, ()>,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let kind = update["sessionUpdate"].as_str().unwrap_or_default();
    if kind != "agent_thought_chunk" && *thinking {
        *thinking = false;
        events.push(StreamEvent::ThinkingEnd);
    }
    match kind {
        "agent_message_chunk" => {
            if let Some(t) = content_text(&update["content"]) {
                events.push(StreamEvent::TextDelta(t));
            }
        }
        "agent_thought_chunk" => {
            if let Some(t) = content_text(&update["content"]) {
                if !*thinking {
                    *thinking = true;
                    events.push(StreamEvent::ThinkingStart);
                }
                events.push(StreamEvent::ThinkingDelta(t));
            }
        }
        "tool_call" | "tool_call_update" => {
            let Some(id) = update["toolCallId"].as_str().map(str::to_string) else {
                return events;
            };
            if kind == "tool_call" && !open_tools.contains_key(&id) {
                open_tools.insert(id.clone(), ());
                let name = update["kind"]
                    .as_str()
                    .filter(|k| !k.is_empty() && *k != "other")
                    .or_else(|| update["title"].as_str())
                    .unwrap_or("tool")
                    .to_string();
                let mut input = update
                    .get("rawInput")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                if let (Some(obj), Some(title)) = (input.as_object_mut(), update["title"].as_str()) {
                    obj.entry("title").or_insert_with(|| json!(title));
                }
                events.push(StreamEvent::ToolUseStart {
                    id: id.clone(),
                    name,
                });
                events.push(StreamEvent::ToolInputDelta(input.to_string()));
                events.push(StreamEvent::ToolUseEnd);
            }
            let status = update["status"].as_str().unwrap_or_default();
            if (status == "completed" || status == "failed") && open_tools.remove(&id).is_some() {
                events.push(StreamEvent::ToolResult {
                    tool_use_id: id,
                    content: tool_output_text(update),
                    is_error: status == "failed",
                });
            }
        }
        _ => {}
    }
    events
}

#[async_trait]
impl Provider for AntigravityCliProvider {
    async fn complete(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> Result<EventStream> {
        let model = self.model();
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
        let mut guard = self.session.lock().await;
        let fresh = guard.is_none();
        if fresh {
            let conn = AcpConnection::spawn(&cwd).await?;
            *guard = Some(Self::open_session(conn, &cwd).await?);
        }
        let session = guard.as_mut().expect("session just opened");
        if let Err(err) = Self::ensure_model(session, &model).await {
            *guard = None;
            return Err(err);
        }
        let prompt_text = build_turn_prompt(messages, fresh);
        if prompt_text.is_empty() {
            bail!("nothing to send to Antigravity CLI");
        }

        let conn = session.conn.clone();
        let session_id = session.session_id.clone();
        drop(guard);

        let (update_tx, mut update_rx) = mpsc::unbounded_channel();
        *conn.updates.lock().unwrap() = Some(update_tx);
        let (_, mut done_rx, msg) = conn.start_request(
            "session/prompt",
            json!({"sessionId": session_id, "prompt": [{"type": "text", "text": prompt_text}]}),
        );
        if let Err(err) = conn.write(&msg).await {
            *self.session.lock().await = None;
            return Err(err.context("failed to send prompt to agy acp"));
        }

        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>(256);
        let session_slot = self.session.clone();
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(StreamEvent::ConnectionType {
                    connection: "Antigravity CLI (ACP)".to_string(),
                }))
                .await;
            let mut thinking = false;
            let mut open_tools = HashMap::new();
            let mut cancelled = false;
            let outcome = loop {
                tokio::select! {
                    Some(params) = update_rx.recv() => {
                        if params["sessionId"].as_str() != Some(session_id.as_str()) {
                            continue;
                        }
                        for event in translate_update(&params["update"], &mut thinking, &mut open_tools) {
                            if tx.send(Ok(event)).await.is_err() && !cancelled {
                                cancelled = true;
                                let _ = conn.notify("session/cancel", json!({"sessionId": session_id})).await;
                            }
                        }
                    }
                    result = &mut done_rx => break result,
                }
            };
            // Drain updates that raced the prompt response.
            while let Ok(params) = update_rx.try_recv() {
                for event in translate_update(&params["update"], &mut thinking, &mut open_tools) {
                    let _ = tx.send(Ok(event)).await;
                }
            }
            if thinking {
                let _ = tx.send(Ok(StreamEvent::ThinkingEnd)).await;
            }
            conn.updates.lock().unwrap().take();
            match outcome {
                Ok(Ok(result)) => {
                    let stop = result["stopReason"].as_str().unwrap_or("end_turn").to_string();
                    if let Some(usage) = result.get("usage").filter(|u| u.is_object()) {
                        let _ = tx
                            .send(Ok(StreamEvent::TokenUsage {
                                input_tokens: usage["inputTokens"].as_u64(),
                                output_tokens: usage["outputTokens"].as_u64(),
                                cache_read_input_tokens: usage["cachedReadTokens"].as_u64(),
                                cache_creation_input_tokens: usage["cachedWriteTokens"].as_u64(),
                            }))
                            .await;
                    }
                    let _ = tx
                        .send(Ok(StreamEvent::MessageEnd {
                            stop_reason: Some(stop),
                        }))
                        .await;
                }
                Ok(Err(err)) => {
                    session_slot.lock().await.take();
                    let _ = tx.send(Err(err)).await;
                }
                Err(_) => {
                    session_slot.lock().await.take();
                    let _ = tx
                        .send(Err(anyhow!("Antigravity CLI (agy acp) exited mid-turn")))
                        .await;
                }
            }
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> String {
        "Antigravity CLI".to_string()
    }

    fn model(&self) -> String {
        self.model.read().unwrap().clone()
    }

    fn set_model(&self, model: &str) -> Result<()> {
        let model = model
            .strip_prefix("antigravity-cli:")
            .unwrap_or(model)
            .trim();
        if model.is_empty() {
            bail!("empty model id");
        }
        *self.model.write().unwrap() = model.to_string();
        Ok(())
    }

    fn available_models(&self) -> Vec<&'static str> {
        FALLBACK_MODELS.to_vec()
    }

    fn available_models_display(&self) -> Vec<String> {
        let mut models = vec![DEFAULT_MODEL.to_string()];
        models.extend(discovered_model_ids());
        models
    }

    fn available_models_for_switching(&self) -> Vec<String> {
        self.available_models_display()
    }

    async fn prefetch_models(&self) -> Result<()> {
        if !DISCOVERED_MODELS.read().unwrap().is_empty() || !cli_available() {
            return Ok(());
        }
        self.refresh_models_now().await
    }

    fn active_auth_method_label(&self) -> Option<&'static str> {
        Some("Antigravity CLI login (via ACP)")
    }

    fn handles_tools_internally(&self) -> bool {
        true
    }

    fn supports_compaction(&self) -> bool {
        false
    }

    fn transport(&self) -> Option<String> {
        Some("ACP (agy acp, stdio)".to_string())
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(Self::with_model(&self.model()))
    }
}

impl AntigravityCliProvider {
    /// Force a live catalog fetch from the CLI, keeping the last good list on failure.
    pub async fn refresh_models_now(&self) -> Result<()> {
        let cwd = std::env::temp_dir();
        let conn = AcpConnection::spawn(&cwd).await?;
        conn.initialize().await?;
        let created = conn
            .request("session/new", json!({"cwd": cwd.to_string_lossy(), "mcpServers": []}))
            .await?;
        let models = parse_models(&created);
        if models.is_empty() {
            bail!("Antigravity CLI returned an empty model list");
        }
        store_models(models);
        if let Some(sid) = created["sessionId"].as_str() {
            let _ = conn.request("session/close", json!({"sessionId": sid})).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
