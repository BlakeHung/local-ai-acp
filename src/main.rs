//! acp-bridge — ACP adapter for local AI services.
//!
//! Bridges any OpenAI-compatible API (Ollama, LocalAI, vLLM, llama.cpp,
//! LM Studio, text-generation-webui) to Agent Client Protocol (ACP).
//!
//! Reads JSON-RPC 2.0 from stdin, translates to HTTP chat completions,
//! and writes JSON-RPC notifications/responses to stdout.
//!
//! Compatible with openab and any ACP-compliant harness.

use acp_bridge::acp;
use acp_bridge::config::ConfigFile;
use acp_bridge::llm;
use acp_bridge::protocol::{AcpError, JsonRpcRequest, Session};
use acp_bridge::tools;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Maximum number of tool call rounds to prevent infinite loops.
const MAX_TOOL_ROUNDS: usize = 5;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static SESSIONS: std::sync::LazyLock<RwLock<HashMap<String, Session>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

// ---------------------------------------------------------------------------
// Session helpers
// ---------------------------------------------------------------------------

fn sessions_write() -> std::sync::RwLockWriteGuard<'static, HashMap<String, Session>> {
    match SESSIONS.write() {
        Ok(s) => s,
        Err(p) => {
            warn!("Session lock poisoned, recovering");
            p.into_inner()
        }
    }
}

fn sessions_read() -> std::sync::RwLockReadGuard<'static, HashMap<String, Session>> {
    match SESSIONS.read() {
        Ok(s) => s,
        Err(p) => {
            warn!("Session lock poisoned, recovering");
            p.into_inner()
        }
    }
}

/// Evict sessions that have been idle longer than the timeout.
fn evict_idle_sessions(timeout_secs: u64) {
    if timeout_secs == 0 {
        return;
    }
    let timeout = Duration::from_secs(timeout_secs);
    let mut sessions = sessions_write();
    let before = sessions.len();
    sessions.retain(|_id, session| session.last_active.elapsed() < timeout);
    let evicted = before - sessions.len();
    if evicted > 0 {
        info!(evicted, remaining = sessions.len(), "Evicted idle sessions");
    }
}

// ---------------------------------------------------------------------------
// ACP method handlers
// ---------------------------------------------------------------------------

fn handle_initialize(id: u64, config: &llm::LlmConfig) {
    info!(model = %config.model, base_url = %config.base_url, "Initialize");
    acp::send_response(
        id,
        json!({
            "agentInfo": {
                "name": format!("acp-bridge ({})", config.model),
                "version": env!("CARGO_PKG_VERSION")
            },
            "capabilities": {}
        }),
    );
}

fn handle_session_new(id: u64, params: &Value, config: &llm::LlmConfig) {
    // Enforce max_sessions limit
    if config.max_sessions > 0 {
        let count = sessions_read().len();
        if count >= config.max_sessions {
            let err = AcpError::SessionLimitReached {
                max: config.max_sessions,
            };
            acp::send_error(id, err.code(), &err.to_string());
            return;
        }
    }

    let raw_cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or("/tmp");

    // Sanitize cwd: only allow typical path characters to prevent prompt injection.
    let cwd: String = raw_cwd
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '/' | '-' | '_' | '.' | ' ' | '~'))
        .collect();

    let session_id = Uuid::new_v4().to_string();

    let system_prompt = std::env::var("LLM_SYSTEM_PROMPT").unwrap_or_else(|_| {
        format!("You are a helpful coding assistant. The user's working directory is: {cwd}")
    });

    let session = Session::new(
        json!({"role": "system", "content": system_prompt}),
        PathBuf::from(&cwd),
    );
    sessions_write().insert(session_id.clone(), session);

    info!(session_id = %session_id, max_history = config.max_history_turns, "New session");
    acp::send_response(id, json!({"sessionId": session_id}));
}

async fn handle_session_prompt(id: u64, params: &Value, config: &llm::LlmConfig) {
    let session_id = match params.get("sessionId").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            let err = AcpError::MissingParam {
                field: "sessionId".into(),
            };
            acp::send_error(id, err.code(), &err.to_string());
            return;
        }
    };

    let user_text = params
        .get("prompt")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    // Add user message, touch session, and trim history
    {
        let mut sessions = sessions_write();
        let session = match sessions.get_mut(&session_id) {
            Some(s) => s,
            None => {
                let err = AcpError::UnknownSession {
                    session_id: session_id.clone(),
                };
                acp::send_error(id, err.code(), &err.to_string());
                return;
            }
        };
        session.touch();
        session
            .messages
            .push(json!({"role": "user", "content": user_text}));

        if config.max_history_turns > 0 {
            let before = session.messages.len();
            session.trim_history(config.max_history_turns);
            let after = session.messages.len();
            if before != after {
                debug!(before, after, "Trimmed conversation history");
            }
        }
    }

    acp::notify_thinking();
    acp::notify_tool_start("llm_chat");

    let mut had_error = false;
    let tool_defs = tools::tool_definitions();

    // Tool call loop: LLM may request tools, we execute and feed results back
    for round in 0..MAX_TOOL_ROUNDS {
        let messages = {
            let sessions = sessions_read();
            sessions
                .get(&session_id)
                .map(|s| s.messages.clone())
                .unwrap_or_default()
        };

        let working_dir = {
            let sessions = sessions_read();
            sessions
                .get(&session_id)
                .map(|s| s.working_dir.clone())
                .unwrap_or_default()
        };

        // Try non-streaming first to check for tool calls
        let chat_result = llm::chat(config, &messages, None, Some(&tool_defs)).await;

        match chat_result {
            Ok(response) => {
                // Check for tool calls in response
                let tool_calls = extract_tool_calls(&response);

                if tool_calls.is_empty() {
                    // No tool calls — extract text and stream it
                    let text = extract_response_text(&response);
                    if !text.is_empty() {
                        acp::notify_text(&text);
                        let mut sessions = sessions_write();
                        if let Some(session) = sessions.get_mut(&session_id) {
                            session
                                .messages
                                .push(json!({"role": "assistant", "content": text}));
                        }
                    }
                    break;
                }

                // Has tool calls — execute them
                info!(round, count = tool_calls.len(), "Executing tool calls");

                // Add assistant message with tool_calls to history
                {
                    let mut sessions = sessions_write();
                    if let Some(session) = sessions.get_mut(&session_id) {
                        let assistant_msg = if config.is_ollama_native() {
                            json!({"role": "assistant", "content": "", "tool_calls": tool_calls})
                        } else {
                            // OpenAI format
                            response["choices"][0]["message"].clone()
                        };
                        session.messages.push(assistant_msg);
                    }
                }

                for tc in &tool_calls {
                    let func = &tc["function"];
                    let name = func["name"].as_str().unwrap_or("unknown");
                    let args_str = func["arguments"].as_str().unwrap_or("{}");
                    let args: Value = serde_json::from_str(args_str)
                        .unwrap_or_else(|_| func["arguments"].clone());

                    acp::notify_tool_start(name);
                    let result = tools::execute_tool(&working_dir, name, &args);
                    acp::notify_tool_done(name, "completed");

                    debug!(tool = name, result_len = result.len(), "Tool executed");

                    // Add tool result to history
                    {
                        let mut sessions = sessions_write();
                        if let Some(session) = sessions.get_mut(&session_id) {
                            session.messages.push(json!({
                                "role": "tool",
                                "content": result
                            }));
                        }
                    }
                }

                // Continue loop — LLM will see tool results and respond
            }
            Err(e) => {
                let err = AcpError::LlmError { reason: e };
                error!(error = %err, "LLM communication failed");
                acp::notify_text(&format!("\n\n**Error:** {err}\n"));
                had_error = true;
                break;
            }
        }
    }

    let status = if had_error { "failed" } else { "completed" };
    acp::notify_tool_done("llm_chat", status);
    acp::send_response(id, json!({"status": status}));
}

/// Extract tool calls from an LLM response (supports both Ollama and OpenAI format).
fn extract_tool_calls(response: &Value) -> Vec<Value> {
    // Ollama native: response.message.tool_calls
    if let Some(calls) = response
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
    {
        return calls.clone();
    }

    // OpenAI compat: response.choices[0].message.tool_calls
    if let Some(calls) = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
    {
        return calls.clone();
    }

    vec![]
}

/// Extract text content from an LLM response (supports both formats).
fn extract_response_text(response: &Value) -> String {
    // Ollama native: response.message.content
    if let Some(text) = response
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        if !text.is_empty() {
            return text.to_string();
        }
    }

    // OpenAI compat: response.choices[0].message.content
    if let Some(text) = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        return text.to_string();
    }

    String::new()
}

fn handle_session_end(id: u64, params: &Value) {
    let session_id = match params.get("sessionId").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            let err = AcpError::MissingParam {
                field: "sessionId".into(),
            };
            acp::send_error(id, err.code(), &err.to_string());
            return;
        }
    };

    let removed = sessions_write().remove(&session_id).is_some();

    if removed {
        info!(session_id = %session_id, "Session ended");
        acp::send_response(id, json!({"status": "ended"}));
    } else {
        let err = AcpError::UnknownSession { session_id };
        acp::send_error(id, err.code(), &err.to_string());
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Handle --version / -V before anything else
    if let Some(arg) = std::env::args().nth(1) {
        if arg == "--version" || arg == "-V" {
            println!("acp-bridge {}", env!("CARGO_PKG_VERSION"));
            return;
        }
    }

    // Initialize tracing — writes to stderr, respects RUST_LOG env.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "acp_bridge=info".parse().unwrap()),
        )
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();

    // Load config: CLI arg (optional TOML path) → env vars → defaults
    let config_path = std::env::args().nth(1);
    let config = match config_path {
        Some(path) => {
            let file = ConfigFile::load(std::path::Path::new(&path));
            file.into_llm_config()
        }
        None => llm::LlmConfig::from_env(),
    };
    info!(
        version = env!("CARGO_PKG_VERSION"),
        model = %config.model,
        base_url = %config.base_url,
        ollama_native = config.is_ollama_native(),
        max_history_turns = config.max_history_turns,
        max_sessions = config.max_sessions,
        session_idle_timeout_secs = config.session_idle_timeout_secs,
        "Starting acp-bridge"
    );

    // Probe backend
    match llm::probe_backend(&config).await {
        Ok(models) if models.is_empty() => {
            info!("Connected to backend (no models listed)");
        }
        Ok(models) => {
            info!(count = models.len(), "Available models:");
            for m in &models {
                info!("  - {m}");
            }
            if !models.iter().any(|m| {
                m.starts_with(&config.model)
                    || config.model.starts_with(m.split(':').next().unwrap_or(""))
            }) {
                warn!(configured = %config.model, "Configured model not found in available models");
            }
        }
        Err(reason) => {
            warn!(
                base_url = %config.base_url,
                error = %reason,
                "Cannot reach backend — will retry on first request"
            );
        }
    }

    // Query model info (Ollama native only)
    if let Some(info) = llm::query_model_info(&config).await {
        info!(
            context_length = info.context_length,
            "Model info from /api/show"
        );
    }

    // Check running models (Ollama)
    if let Some(running) = llm::query_running_models(&config).await {
        if running.is_empty() {
            warn!(
                model = %config.model,
                "No models loaded in VRAM — first request may be slow. Run: ollama run {}",
                config.model
            );
        } else {
            info!(count = running.len(), "Running models (loaded in VRAM):");
            for m in &running {
                info!("  - {m}");
            }
        }
    }

    // Spawn idle session cleanup task
    let idle_timeout = config.session_idle_timeout_secs;
    if idle_timeout > 0 {
        tokio::spawn(async move {
            let interval = Duration::from_secs(idle_timeout.min(60));
            loop {
                tokio::time::sleep(interval).await;
                evict_idle_sessions(idle_timeout);
            }
        });
    }

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    loop {
        tokio::select! {
            line_result = lines.next_line() => {
                match line_result {
                    Ok(Some(line)) => {
                        let trimmed = line.trim().to_string();
                        if trimmed.is_empty() {
                            continue;
                        }

                        let msg: JsonRpcRequest = match serde_json::from_str(&trimmed) {
                            Ok(m) => m,
                            Err(e) => {
                                debug!(error = %e, "Skipping invalid JSON-RPC line");
                                continue;
                            }
                        };

                        let id = msg.id;
                        let method = msg.method.as_str();
                        let params = msg.params.clone().unwrap_or(json!({}));

                        debug!(id, method, "Received request");

                        match method {
                            "initialize" => handle_initialize(id, &config),
                            "session/new" => handle_session_new(id, &params, &config),
                            "session/prompt" => handle_session_prompt(id, &params, &config).await,
                            "session/end" => handle_session_end(id, &params),
                            _ => {
                                let err = AcpError::MethodNotFound { method: method.to_string() };
                                acp::send_error(id, err.code(), &err.to_string());
                            }
                        }
                    }
                    Ok(None) => {
                        info!("stdin closed, shutting down gracefully");
                        break;
                    }
                    Err(e) => {
                        error!(error = %e, "Error reading stdin");
                        break;
                    }
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Received shutdown signal, exiting");
                break;
            }
        }
    }

    // Cleanup
    let session_count = {
        let mut s = sessions_write();
        let n = s.len();
        s.clear();
        n
    };
    if session_count > 0 {
        info!(sessions = session_count, "Cleaned up sessions on exit");
    }
}
