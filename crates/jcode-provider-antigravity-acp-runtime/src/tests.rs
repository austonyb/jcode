use super::*;

#[test]
fn discovers_arbitrary_future_model_from_session_new() {
    let response = json!({
        "sessionId": "s1",
        "models": {"availableModels": [
            {"modelId": "gemini-99-future-xhigh", "name": "Gemini 99 Future"},
            {"modelId": "claude-opus-5-5-high", "name": "Claude Opus 5.5 (High)"}
        ]}
    });
    let models = parse_models(&response);
    assert_eq!(models[0].0, "gemini-99-future-xhigh");
    assert!(!FALLBACK_MODELS.contains(&"gemini-99-future-xhigh"));
    // Empty response keeps the last good catalog.
    store_models(models);
    store_models(parse_models(&json!({})));
    assert!(discovered_model_ids().contains(&"gemini-99-future-xhigh".to_string()));
}

#[test]
fn permission_requests_pick_allow_once() {
    let req = json!({"jsonrpc":"2.0","id":7,"method":"session/request_permission","params":{
        "options":[{"optionId":"no","kind":"reject_once"},{"optionId":"yes","kind":"allow_once"}]}});
    let reply = handle_agent_request(&req);
    assert_eq!(reply["id"], 7);
    assert_eq!(reply["result"]["outcome"]["optionId"], "yes");
    let fs = json!({"jsonrpc":"2.0","id":8,"method":"fs/read_text_file","params":{}});
    assert!(handle_agent_request(&fs).get("error").is_some());
}

#[test]
fn translates_text_thought_and_tool_updates() {
    let mut thinking = false;
    let mut tools = HashMap::new();
    let ev = translate_update(
        &json!({"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"hm"}}),
        &mut thinking,
        &mut tools,
    );
    assert!(matches!(ev[0], StreamEvent::ThinkingStart));
    let ev = translate_update(
        &json!({"sessionUpdate":"tool_call","toolCallId":"t1","title":"Read foo","kind":"read","status":"pending"}),
        &mut thinking,
        &mut tools,
    );
    assert!(matches!(ev[0], StreamEvent::ThinkingEnd));
    assert!(matches!(&ev[1], StreamEvent::ToolUseStart { id, name } if id == "t1" && name == "read"));
    let ev = translate_update(
        &json!({"sessionUpdate":"tool_call_update","toolCallId":"t1","status":"completed",
            "content":[{"type":"content","content":{"type":"text","text":"ok"}}]}),
        &mut thinking,
        &mut tools,
    );
    assert!(matches!(&ev[0], StreamEvent::ToolResult { content, is_error: false, .. } if content == "ok"));
    let ev = translate_update(
        &json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hi"}}),
        &mut thinking,
        &mut tools,
    );
    assert!(matches!(&ev[0], StreamEvent::TextDelta(t) if t == "hi"));
}

fn msg(role: Role, text: &str) -> Message {
    Message {
        role,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
            cache_control: None,
        }]
        .into(),
        timestamp: None,
        tool_duration_ms: None,
    }
}

#[test]
fn prompt_sends_only_new_turn_on_live_session_and_replays_on_fresh() {
    let history = vec![
        msg(Role::User, "first"),
        msg(Role::Assistant, "answer"),
        msg(Role::User, "second"),
    ];
    assert_eq!(build_turn_prompt(&history, false), "second");
    let fresh = build_turn_prompt(&history, true);
    assert!(fresh.contains("<previous_conversation>"));
    assert!(fresh.contains("answer"));
    assert!(fresh.ends_with("second"));
}
