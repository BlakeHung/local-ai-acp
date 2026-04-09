//! ACP JSON-RPC helpers — stdout transport and notification builders.

use serde_json::{json, Value};
use std::io::Write;

/// Write a JSON-RPC object to stdout (newline-delimited).
pub fn send(obj: &Value) {
    let mut stdout = std::io::stdout().lock();
    let _ = serde_json::to_writer(&mut stdout, obj);
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
}

/// Send a JSON-RPC success response.
pub fn send_response(id: u64, result: Value) {
    send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
}

/// Send a JSON-RPC error response.
pub fn send_error(id: u64, code: i64, message: &str) {
    send(&json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}));
}

/// Send a JSON-RPC notification (no id).
pub fn send_notification(method: &str, params: Value) {
    send(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
}

/// Notify an agent_message_chunk (streaming text).
pub fn notify_text(text: &str) {
    send_notification(
        "session/notify",
        json!({"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": text}}}),
    );
}

/// Notify an agent_thought_chunk.
pub fn notify_thinking() {
    send_notification(
        "session/notify",
        json!({"update": {"sessionUpdate": "agent_thought_chunk"}}),
    );
}

/// Notify a tool_call start.
pub fn notify_tool_start(title: &str) {
    send_notification(
        "session/notify",
        json!({"update": {"sessionUpdate": "tool_call", "title": title}}),
    );
}

/// Notify a tool_call_update (completion).
pub fn notify_tool_done(title: &str, status: &str) {
    send_notification(
        "session/notify",
        json!({"update": {"sessionUpdate": "tool_call_update", "title": title, "status": status}}),
    );
}
