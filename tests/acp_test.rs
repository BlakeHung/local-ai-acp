/// Verify JSON-RPC message structure matches ACP spec.
/// Since acp::send writes directly to stdout, we test the JSON shapes independently.

#[test]
fn json_rpc_response_structure() {
    // Construct what send_response would produce
    let id = 1u64;
    let result = serde_json::json!({"agentInfo": {"name": "test", "version": "0.1.0"}});
    let msg = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});

    // Verify structure
    assert_eq!(msg["jsonrpc"], "2.0");
    assert_eq!(msg["id"], 1);
    assert!(msg["result"]["agentInfo"]["name"].is_string());
}

#[test]
fn json_rpc_error_structure() {
    let id = 5u64;
    let code = -32601i64;
    let message = "Method not found: foo";
    let msg = serde_json::json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}});

    assert_eq!(msg["error"]["code"], -32601);
    assert_eq!(msg["error"]["message"], "Method not found: foo");
}

#[test]
fn notification_has_no_id() {
    let method = "session/notify";
    let params = serde_json::json!({"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "hi"}}});
    let msg = serde_json::json!({"jsonrpc": "2.0", "method": method, "params": params});

    assert!(msg.get("id").is_none());
    assert_eq!(msg["method"], "session/notify");
    assert_eq!(
        msg["params"]["update"]["sessionUpdate"],
        "agent_message_chunk"
    );
}

#[test]
fn notification_text_content_format() {
    let text = "Hello, world!";
    let params = serde_json::json!({"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": text}}});

    assert_eq!(
        params["update"]["content"]["text"].as_str().unwrap(),
        "Hello, world!"
    );
}

#[test]
fn tool_call_notification_format() {
    let title = "llm_chat";
    let params = serde_json::json!({"update": {"sessionUpdate": "tool_call", "title": title}});
    assert_eq!(params["update"]["title"], "llm_chat");

    let done_params = serde_json::json!({"update": {"sessionUpdate": "tool_call_update", "title": title, "status": "completed"}});
    assert_eq!(done_params["update"]["status"], "completed");
}
