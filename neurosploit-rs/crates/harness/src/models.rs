use anyhow::{anyhow, Result};
use serde::Serialize;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

/// A model provider exposing an OpenAI-compatible `/chat/completions` endpoint.
#[derive(Clone, Debug, Serialize)]
pub struct Provider {
    pub key: &'static str,
    pub label: &'static str,
    pub base_url: &'static str,
    pub env_key: &'static str,
    /// "cli" (also drivable by an agentic CLI) | "api"
    pub kind: &'static str,
    pub models: Vec<&'static str>,
}

/// The full provider registry. Every entry speaks the OpenAI chat schema
/// (Anthropic, xAI, NVIDIA NIM, DeepSeek, Mistral, Qwen, Groq, Together,
/// OpenRouter, Gemini-compat, Ollama).
pub fn providers() -> Vec<Provider> {
    vec![
        Provider { key: "anthropic", label: "Anthropic Claude", base_url: "https://api.anthropic.com/v1", env_key: "ANTHROPIC_API_KEY", kind: "cli",
            models: vec!["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"] },
        Provider { key: "openai", label: "OpenAI (ChatGPT)", base_url: "https://api.openai.com/v1", env_key: "OPENAI_API_KEY", kind: "cli",
            models: vec!["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1", "gpt-5.1-codex", "o4"] },
        Provider { key: "xai", label: "xAI Grok", base_url: "https://api.x.ai/v1", env_key: "XAI_API_KEY", kind: "cli",
            models: vec!["grok-4", "grok-4-fast"] },
        Provider { key: "gemini", label: "Google Gemini", base_url: "https://generativelanguage.googleapis.com/v1beta/openai", env_key: "GEMINI_API_KEY", kind: "cli",
            models: vec!["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash"] },
        Provider { key: "nvidia_nim", label: "NVIDIA NIM", base_url: "https://integrate.api.nvidia.com/v1", env_key: "NVIDIA_NIM_API_KEY", kind: "api",
            models: vec!["nvidia/llama-3.3-nemotron-super-49b-v1", "deepseek-ai/deepseek-r1", "qwen/qwen2.5-coder-32b-instruct"] },
        Provider { key: "nvidia", label: "NVIDIA (public API)", base_url: "https://integrate.api.nvidia.com/v1", env_key: "NVIDIA_API_KEY", kind: "api",
            models: vec![
                "nvidia/glm-5.3-flash",
                "nvidia/nemotron-3-ultra",
            ] },
        Provider { key: "deepseek", label: "DeepSeek", base_url: "https://api.deepseek.com/v1", env_key: "DEEPSEEK_API_KEY", kind: "api",
            models: vec!["deepseek-reasoner", "deepseek-chat"] },
        Provider { key: "mistral", label: "Mistral", base_url: "https://api.mistral.ai/v1", env_key: "MISTRAL_API_KEY", kind: "api",
            models: vec!["mistral-large-latest", "codestral-latest"] },
        Provider { key: "qwen", label: "Qwen (DashScope)", base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", env_key: "DASHSCOPE_API_KEY", kind: "api",
            models: vec!["qwen-max", "qwen2.5-coder-32b-instruct", "qwq-plus"] },
        Provider { key: "groq", label: "Groq", base_url: "https://api.groq.com/openai/v1", env_key: "GROQ_API_KEY", kind: "api",
            models: vec!["llama-3.3-70b-versatile", "qwen-2.5-coder-32b"] },
        Provider { key: "together", label: "Together AI", base_url: "https://api.together.xyz/v1", env_key: "TOGETHER_API_KEY", kind: "api",
            models: vec!["Qwen/Qwen2.5-Coder-32B-Instruct", "deepseek-ai/DeepSeek-R1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"] },
        // LiteLLM proxy (OpenAI-compatible). Point at your gateway with
        // LITELLM_BASE_URL (default http://localhost:4000/v1); key = LITELLM_API_KEY.
        // Use `litellm:<any-model-the-proxy-routes>` — model names pass through.
        Provider { key: "litellm", label: "LiteLLM (proxy)", base_url: "http://localhost:4000/v1", env_key: "LITELLM_API_KEY", kind: "api",
            models: vec!["gpt-4o", "claude-3-7-sonnet", "gemini/gemini-2.5-pro"] },
        Provider { key: "openrouter", label: "OpenRouter", base_url: "https://openrouter.ai/api/v1", env_key: "OPENROUTER_API_KEY", kind: "api",
            models: vec!["anthropic/claude-opus-4-8", "qwen/qwen-2.5-coder-32b-instruct", "deepseek/deepseek-r1", "meta-llama/llama-3.3-70b-instruct"] },
        // Azure OpenAI (OpenAI-compatible). Set AZURE_OPENAI_ENDPOINT (e.g.
        // https://<resource>.openai.azure.com), optionally AZURE_OPENAI_API_VERSION
        // (default 2024-10-21), and use `azure:<your-deployment-name>` as the model.
        // base_url is resolved from the endpoint at call time; auth uses an api-key header.
        Provider { key: "azure", label: "Azure OpenAI", base_url: "", env_key: "AZURE_OPENAI_API_KEY", kind: "api",
            models: vec!["gpt-4o", "gpt-4o-mini", "gpt-5.1", "o4-mini"] },
        Provider { key: "ollama", label: "Ollama (local)", base_url: "http://localhost:11434/v1", env_key: "OLLAMA_API_KEY", kind: "api",
            models: vec!["qwen2.5-coder:32b", "qwq:32b", "deepseek-r1:32b", "llama3.3:70b"] },
    ]
}

pub fn provider_for(key: &str) -> Option<Provider> {
    providers().into_iter().find(|p| p.key == key)
}

/// Resolve a provider's API key from the environment, honoring common aliases.
/// For Gemini we also accept `GOOGLE_API_KEY` (Google's standard env var name)
/// when `GEMINI_API_KEY` is unset.
fn resolve_key(p: &Provider) -> String {
    let mut k = std::env::var(p.env_key).unwrap_or_default();
    if k.is_empty() && p.key == "gemini" {
        k = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
    }
    k
}

/// A `provider:model` selection.
#[derive(Clone, Debug)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

impl ModelRef {
    pub fn parse(s: &str) -> ModelRef {
        match s.split_once(':') {
            Some((p, m)) => ModelRef { provider: p.to_string(), model: m.to_string() },
            None => ModelRef { provider: "anthropic".into(), model: s.to_string() },
        }
    }
    pub fn label(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }
}

/// OpenAI-compatible chat client shared across the model pool.
#[derive(Clone)]
pub struct ChatClient {
    http: reqwest::Client,
}

impl ChatClient {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        ChatClient { http }
    }

    /// One chat completion. Errors (missing key, network, non-2xx) propagate so
    /// the pool can fail over to the next candidate model.
    pub async fn chat(&self, m: &ModelRef, system: &str, user: &str) -> Result<String> {
        let p = provider_for(&m.provider)
            .ok_or_else(|| anyhow!("unknown provider '{}'", m.provider))?;
        let key = resolve_key(&p);
        if key.is_empty() && p.key != "ollama" && p.key != "litellm" {
            let hint = if p.key == "gemini" { format!("{} (or GOOGLE_API_KEY)", p.env_key) } else { p.env_key.to_string() };
            return Err(anyhow!("no API key ({}) for provider '{}'", hint, p.key));
        }
        // Azure OpenAI uses a per-resource endpoint + deployment + api-version,
        // and authenticates with an `api-key` header instead of Bearer.
        let azure = p.key == "azure";
        let url = if azure {
            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_default();
            if endpoint.is_empty() {
                return Err(anyhow!("set AZURE_OPENAI_ENDPOINT (e.g. https://<resource>.openai.azure.com) for the azure provider"));
            }
            let ver = std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| "2024-10-21".to_string());
            // `model` is the Azure DEPLOYMENT name (use `azure:<deployment>`).
            format!("{}/openai/deployments/{}/chat/completions?api-version={}",
                endpoint.trim_end_matches('/'), m.model, ver)
        } else {
            // Allow an env base-URL override (LiteLLM gateway, self-hosted proxies, …).
            let base = match p.key {
                "litellm" => std::env::var("LITELLM_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
                "ollama" => std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
                _ => p.base_url.to_string(),
            };
            format!("{}/chat/completions", base.trim_end_matches('/'))
        };
        let body = serde_json::json!({
            "model": m.model,
            "max_tokens": 4096,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        });
        let mut req = self.http.post(&url).json(&body);
        if !key.is_empty() {
            if azure { req = req.header("api-key", &key); } else { req = req.bearer_auth(&key); }
        }
        let resp = req.send().await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("{} returned {}: {}", p.key, status, truncate(&text, 200)));
        }
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("no content in response"))?;
        Ok(content.to_string())
    }
}

impl ChatClient {
    /// Complete via a locally-installed **agentic CLI subscription** (Claude
    /// Code / Codex / Grok / Gemini) instead of an API key. This uses the user's
    /// logged-in subscription, so no provider key is required.
    ///
    /// When `mcp_config` is set (a path to an `.mcp.json`), Claude/Codex run with
    /// the MCP servers enabled and tool autonomy, so agents can actually drive
    /// **Playwright** (browse, execute JS, screenshot) during execution.
    pub async fn chat_cli(
        &self,
        label: &str,
        provider: &str,
        model: &str,
        system: &str,
        user: &str,
        mcp_config: Option<&str>,
        progress: Option<tokio::sync::mpsc::Sender<String>>,
    ) -> Result<String> {
        let bin = cli_binary_for(provider)
            .ok_or_else(|| anyhow!("no CLI/subscription backend for provider '{}'", provider))?;
        let prompt = format!("{}\n\n{}", system, user);

        // Claude Code can stream structured events (tools, commands, files) which
        // we surface live as a categorized activity feed, attributed to `label`.
        if bin == "claude" {
            return self.chat_claude_stream(label, model, &prompt, mcp_config, progress).await;
        }

        let mut cmd = Command::new(bin);
        match bin {
            // Codex non-interactive exec (uses the ChatGPT/Codex login), prompt on stdin.
            "codex" => {
                cmd.arg("exec").arg("--model").arg(model)
                    .arg("--dangerously-bypass-approvals-and-sandbox");
                if let Some(mcp) = mcp_config {
                    cmd.arg("--config").arg(format!("mcp_config_file={}", mcp));
                }
                cmd.arg("-");
            }
            // Google Gemini CLI (uses the Gemini subscription login).
            "gemini" => {
                cmd.arg("-m").arg(model);
            }
            // Grok CLI, prompt on stdin (best-effort flags).
            "grok" => {
                cmd.arg("--model").arg(model);
            }
            _ => {}
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| anyhow!("spawn {} failed: {}", bin, e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
        }
        let out = match tokio::time::timeout(Duration::from_secs(600), child.wait_with_output()).await {
            Ok(r) => r?,
            Err(_) => return Err(anyhow!("{} subscription CLI timed out after 600s", bin)),
        };
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !out.status.success() {
            let detail = if !stderr.trim().is_empty() { stderr.trim().to_string() } else if !stdout.is_empty() { stdout.clone() } else { "no output".to_string() };
            return Err(anyhow!(
                "{} subscription CLI exit {}: {}",
                bin,
                out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                truncate(&detail, 240)
            ));
        }
        if stdout.is_empty() {
            return Err(anyhow!("{} subscription CLI returned empty output", bin));
        }
        Ok(stdout)
    }

    /// Drive Claude Code with `--output-format stream-json` and surface its
    /// activity as a live, categorized feed (states, tools, commands, files).
    /// Tagged events are sent to `progress`; the final assistant text is returned.
    async fn chat_claude_stream(
        &self,
        label: &str,
        model: &str,
        prompt: &str,
        mcp_config: Option<&str>,
        progress: Option<tokio::sync::mpsc::Sender<String>>,
    ) -> Result<String> {
        let mut cmd = Command::new("claude");
        cmd.arg("-p").arg("--model").arg(model)
            .arg("--output-format").arg("stream-json").arg("--verbose")
            .arg("--dangerously-skip-permissions")
            .env("IS_SANDBOX", "1");
        if let Some(mcp) = mcp_config {
            cmd.arg("--mcp-config").arg(mcp);
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        let mut child = cmd.spawn().map_err(|e| anyhow!("spawn claude failed: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
        }
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let mut lines = BufReader::new(stdout).lines();
        let lbl = if label.is_empty() { String::new() } else { format!("@{} ", label) };
        let emit = |s: String| {
            if let Some(tx) = &progress { let _ = tx.try_send(format!("{}{}", lbl, s)); }
        };

        let mut result = String::new();
        let mut had_err = String::new();
        let read = async {
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    match v.get("type").and_then(|t| t.as_str()) {
                        Some("assistant") => {
                            if let Some(content) = v.pointer("/message/content").and_then(|c| c.as_array()) {
                                for blk in content {
                                    match blk.get("type").and_then(|t| t.as_str()) {
                                        Some("text") => {
                                            if let Some(t) = blk.get("text").and_then(|x| x.as_str()) {
                                                let t = t.trim();
                                                if !t.is_empty() { emit(format!("ai: {}", truncate(t, 240))); }
                                            }
                                        }
                                        Some("tool_use") => {
                                            let name = blk.get("name").and_then(|x| x.as_str()).unwrap_or("tool");
                                            let input = blk.get("input");
                                            emit(tool_event(name, input));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        Some("result") => {
                            if let Some(r) = v.get("result").and_then(|x| x.as_str()) { result = r.to_string(); }
                            let ti = v.pointer("/usage/input_tokens").and_then(|x| x.as_u64());
                            let to = v.pointer("/usage/output_tokens").and_then(|x| x.as_u64());
                            if ti.is_some() || to.is_some() { emit(format!("tokens: in={} out={}", ti.unwrap_or(0), to.unwrap_or(0))); }
                            if v.get("is_error").and_then(|x| x.as_bool()).unwrap_or(false) { had_err = v.get("result").and_then(|x| x.as_str()).unwrap_or("error").to_string(); }
                        }
                        _ => {}
                    }
                }
            }
        };
        if tokio::time::timeout(Duration::from_secs(900), read).await.is_err() { return Err(anyhow!("claude stream timed out after 900s")); }
        let _ = child.wait().await;
        if !had_err.is_empty() && result.is_empty() { return Err(anyhow!("claude: {}", truncate(&had_err, 240))); }
        if result.is_empty() { return Err(anyhow!("claude stream produced no result")); }
        Ok(result)
    }
}

/// Categorise a Claude tool_use block into a tagged activity-feed event.
fn tool_event(name: &str, input: Option<&serde_json::Value>) -> String {
    let s = |k: &str| input.and_then(|i| i.get(k)).and_then(|x| x.as_str()).unwrap_or("");
    match name {
        "Bash" => {
            let c = s("command");
            let danger = c.contains("rm -rf") || c.contains("mkfs") || c.contains(":(){")
                || c.contains("dd if=") || c.contains("> /dev/");
            format!("{}: {}", if danger { "danger" } else { "exec" }, truncate(c, 200))
        }
        "Read" => format!("read: {}", s("file_path")),
        "Write" | "Edit" => format!("edit: {}", s("file_path")),
        "Grep" => format!("tool: grep {}", truncate(s("pattern"), 80)),
        "Glob" => format!("tool: glob {}", truncate(s("pattern"), 80)),
        "WebFetch" => format!("net: fetch {}", s("url")),
        n if n.contains("playwright") || n.contains("browser") => {
            let url = s("url");
            format!("net: browser {}{}", n.rsplit('_').next().unwrap_or(n), if url.is_empty() { String::new() } else { format!(" {}", url) })
        }
        other => format!("tool: {}", other),
    }
}
