@@
-        let url = if azure {
-            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_default();
-            if endpoint.is_empty() {
-                return Err(anyhow!("set AZURE_OPENAI_ENDPOINT (e.g. https://<resource>.openai.azure.com) for the azure provider"));
-            }
-            let ver = std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| "2024-10-21".to_string());
-            // `model` is the Azure DEPLOYMENT name (use `azure:<deployment>`).
-            format!("{}/openai/deployments/{}/chat/completions?api-version={}",
-                endpoint.trim_end_matches('/'), m.model, ver)
-        } else {
-            // Allow an env base-URL override (LiteLLM gateway, self-hosted proxies, …).
-            let base = match p.key {
-                "litellm" => std::env::var("LITELLM_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
-                "ollama" => std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
-                _ => p.base_url.to_string(),
-            };
-            format!("{}/chat/completions", base.trim_end_matches('/'))
-        };
-        let body = serde_json::json!({
-            "model": m.model,
-            "max_tokens": 4096,
-            "temperature": 0.2,
-            "messages": [
-                {"role": "system", "content": system},
-                {"role": "user", "content": user}
-            ]
-        });
-        let mut req = self.http.post(&url).json(&body);
-        if !key.is_empty() {
-            if azure { req = req.header("api-key", &key); } else { req = req.bearer_auth(&key); }
-        }
-        let resp = req.send().await?;
-        let status = resp.status();
-        let text = resp.text().await.unwrap_or_default();
-        if !status.is_success() {
-            return Err(anyhow!("{} returned {}: {}", p.key, status, truncate(&text, 200)));
-        }
-        let v: serde_json::Value = serde_json::from_str(&text)?;
-        let content = v["choices"][0]["message"]["content"]
-            .as_str()
-            .ok_or_else(|| anyhow!("no content in response"))?;
-        Ok(content.to_string())
+        // Build candidate endpoints (supports NVIDIA public API fallback).
+        let base = match p.key {
+            "litellm" => std::env::var("LITELLM_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
+            "ollama" => std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| p.base_url.to_string()),
+            _ => p.base_url.to_string(),
+        };
+        let base = base.trim_end_matches('/').to_string();
+
+        let body = serde_json::json!({
+            "model": m.model,
+            "max_tokens": max_tokens_for(&m.model),
+            "temperature": 0.2,
+            "messages": [
+                {"role": "system", "content": system},
+                {"role": "user", "content": user}
+            ]
+        });
+
+        let mut candidates: Vec<String> = Vec::new();
+        if azure {
+            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_default();
+            if endpoint.is_empty() {
+                return Err(anyhow!("set AZURE_OPENAI_ENDPOINT (e.g. https://<resource>.openai.azure.com) for the azure provider"));
+            }
+            let ver = std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| "2024-10-21".to_string());
+            candidates.push(format!("{}/openai/deployments/{}/chat/completions?api-version={}",
+                endpoint.trim_end_matches('/'), m.model, ver));
+        } else if p.key == "nvidia" {
+            candidates.push(format!("{}/chat/completions", base));
+            candidates.push(format!("{}/models/{}/chat/completions", base, m.model));
+        } else {
+            candidates.push(format!("{}/chat/completions", base));
+        }
+
+        let mut last_err: Option<anyhow::Error> = None;
+        let mut resp_text = String::new();
+        let mut tried_urls: Vec<String> = Vec::new();
+
+        for url in candidates {
+            tried_urls.push(url.clone());
+            let mut req = self.http.post(&url).json(&body);
+            if !key.is_empty() {
+                if azure { req = req.header("api-key", &key); } else { req = req.bearer_auth(&key); }
+            }
+            match req.send().await {
+                Ok(resp) => {
+                    let status = resp.status();
+                    let text = resp.text().await.unwrap_or_default();
+                    if status.is_success() {
+                        resp_text = text;
+                        break;
+                    } else if status.as_u16() == 404 {
+                        last_err = Some(anyhow!("{} returned 404 at {} — trying fallback", p.key, url));
+                        continue;
+                    } else {
+                        return Err(anyhow!("{} returned {}: {}", p.key, status, truncate(&text, 200)));
+                    }
+                }
+                Err(e) => {
+                    last_err = Some(anyhow!("request to {} failed: {}", url, e));
+                    continue;
+                }
+            }
+        }
+
+        if resp_text.trim().is_empty() {
+            if let Some(e) = last_err {
+                return Err(anyhow!("{} — tried URLs: {:?}", e, tried_urls));
+            }
+            return Err(anyhow!("{} returned empty body for model '{}'; tried: {:?}", p.key, m.model, tried_urls));
+        }
+
+        let v: serde_json::Value = match serde_json::from_str(&resp_text) {
+            Ok(v) => v,
+            Err(e) => {
+                if let Some(sse) = parse_sse_content(&resp_text) {
+                    if !sse.trim().is_empty() {
+                        return Ok(sse);
+                    }
+                }
+                return Err(anyhow!(
+                    "{} returned unparseable body ({} bytes, model '{}'): {} — body starts: {}",
+                    p.key, resp_text.len(), m.model, e, truncate(resp_text.trim_start(), 300)
+                ));
+            }
+        };
+        let choice = &v["choices"][0];
+        if choice.is_null() {
+            let err = v["error"]["message"].as_str().unwrap_or("empty choices");
+            return Err(anyhow!("{} returned no choices: {}", p.key, truncate(err, 200)));
+        }
+        let content = extract_message_text(&choice["message"]);
+        if content.trim().is_empty() {
+            let finish = choice["finish_reason"].as_str().unwrap_or("unknown");
+            let hint = if finish == "length" { " — raised max_tokens may help, or the model is reasoning-heavy for this prompt size" } else { "" };
+            return Err(anyhow!(
+                "{} returned empty content (finish_reason={finish}, model='{}'){hint}",
+                p.key, m.model
+            ));
+        }
+        Ok(content)
*** End Patch