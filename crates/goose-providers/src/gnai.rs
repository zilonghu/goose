use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::Tool;
use serde::Deserialize;
use serde_json::Value;

use crate::anthropic::{AnthropicProvider, AnthropicProviderBuilder, ANTHROPIC_API_VERSION};
use crate::api_client::ApiClient;
use crate::base::{
    ConfigKey, MessageStream, ModelInfo, Provider, ProviderDescriptor, ProviderMetadata,
};
use crate::conversation::message::Message;
use crate::errors::ProviderError;
use crate::model::ModelConfig;
use crate::openai::{OpenAiProvider, OpenAiProviderBuilder};
use crate::openai_compatible::{handle_response_openai_compat, OpenAiCompatibleProvider};

const EFFORT_HEADER: &str = "x-goose-gnai-effective-effort";

#[derive(Clone, Debug, Deserialize)]
pub struct Deployment {
    pub name: String,
    #[serde(default)]
    aliases: Vec<String>,
    capabilities: Vec<String>,
    #[serde(default)]
    extra: HashMap<String, Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Messages,
    Responses,
    Completions,
}

impl Deployment {
    fn route(&self) -> Result<Route, ProviderError> {
        let supports = |capability| self.capabilities.iter().any(|c| c == capability);
        if !supports("chat") {
            return Err(ProviderError::InvalidValue(format!(
                "{} is not a chat model",
                self.name
            )));
        }
        if supports("anthropic-messages") {
            Ok(Route::Messages)
        } else if supports("openai-responses") {
            Ok(Route::Responses)
        } else if supports("openai-completions") {
            Ok(Route::Completions)
        } else {
            Err(ProviderError::InvalidValue(format!(
                "{} advertises no supported protocol",
                self.name
            )))
        }
    }

    fn info(&self) -> ModelInfo {
        let mut info = ModelInfo::new(&self.name);
        info.reasoning = self
            .extra
            .get("reasoning_effort")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| ModelConfig::new(&self.name).is_reasoning_model());
        info
    }

    fn config(&self, requested: &ModelConfig) -> Result<ModelConfig, ProviderError> {
        let mut config = ModelConfig::new(&self.name).with_canonical_limits("gnai");
        // Canonicalization supplies capabilities only; deployments retain their exact identities.
        if requested.model_name == self.name || self.aliases.contains(&requested.model_name) {
            config.model_name.clone_from(&requested.model_name);
        } else {
            config.model_name.clone_from(&self.name);
        }
        config.temperature = requested.temperature;
        config
            .request_headers
            .clone_from(&requested.request_headers);
        let mut params = HashMap::new();
        let effort = requested
            .request_params
            .as_ref()
            .and_then(|p| p.get("thinking_effort"))
            .filter(|_| {
                requested.model_name == self.name || self.aliases.contains(&requested.model_name)
            })
            .or_else(|| self.extra.get("effort"));
        if self.extra.get("reasoning_effort") == Some(&Value::Bool(false)) {
            config.reasoning = Some(false);
        } else if let Some(effort) = effort {
            params.insert("thinking_effort".into(), effort.clone());
        }
        config.request_params = (!params.is_empty()).then_some(params);
        if let Some(effort) = effort.and_then(Value::as_str) {
            config
                .request_headers
                .get_or_insert_default()
                .insert(EFFORT_HEADER.into(), effort.into());
        }
        config.reasoning = self
            .extra
            .get("reasoning_effort")
            .and_then(Value::as_bool)
            .or(config.reasoning);
        Ok(config)
    }

    fn apply_wire_controls(&self, payload: &mut Value, effective_effort: Option<&str>) {
        if self.extra.get("reasoning_effort") == Some(&Value::Bool(true))
            && self.route() != Ok(Route::Messages)
        {
            if let Some(effort) = effective_effort.filter(|effort| *effort != "off") {
                if payload.get("input").is_some() {
                    payload["reasoning"]["effort"] = Value::String(effort.into());
                } else {
                    payload["reasoning_effort"] = Value::String(effort.into());
                }
            }
        }
        if self.extra.get("reasoning_effort") == Some(&Value::Bool(false))
            || effective_effort == Some("off")
        {
            payload.as_object_mut().unwrap().remove("reasoning");
            payload.as_object_mut().unwrap().remove("reasoning_effort");
            payload.as_object_mut().unwrap().remove("thinking");
            payload.as_object_mut().unwrap().remove("output_config");
        } else if self.extra.get("adaptive_thinking") == Some(&Value::Bool(true))
            && payload.get("messages").is_some()
        {
            let effort = effective_effort
                .map(|effort| Value::String(effort.into()))
                .or_else(|| payload.pointer("/output_config/effort").cloned())
                .or_else(|| self.extra.get("effort").cloned());
            payload["thinking"] = serde_json::json!({"type": "adaptive"});
            if let Some(effort) = effort {
                payload["output_config"] = serde_json::json!({"effort": effort});
            }
        }
    }
}

pub struct GnaiProvider {
    catalog: Vec<Deployment>,
    metadata_client: ApiClient,
    chat: OpenAiCompatibleProvider,
    responses: OpenAiProvider,
    messages: AnthropicProvider,
}

impl ProviderDescriptor for GnaiProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "gnai",
            "GNAI",
            "GNAI deployments using their advertised native protocols",
            "",
            vec![],
            "",
            vec![
                ConfigKey::new("GNAI_URL", true, false, None, true),
                ConfigKey::new("GNAI_TOKEN", false, true, None, false),
            ],
        )
    }
}

impl GnaiProvider {
    pub async fn create(
        metadata_client: ApiClient,
        chat_client: ApiClient,
        responses_client: ApiClient,
        messages_client: ApiClient,
    ) -> Result<Self> {
        #[derive(Deserialize)]
        struct Catalog {
            models: Vec<Deployment>,
        }
        let response = metadata_client.response_get("meta").await?;
        let metadata = handle_response_openai_compat(response).await?;
        let catalog: Catalog = serde_json::from_value(metadata)?;
        let catalog: Vec<Deployment> = catalog
            .models
            .into_iter()
            .filter(|m| m.route().is_ok())
            .collect();
        let wire_catalog = catalog.clone();
        let controls = Arc::new(
            move |request: reqwest::RequestBuilder| -> Result<reqwest::RequestBuilder> {
                let mut built = request
                    .try_clone()
                    .ok_or_else(|| anyhow::anyhow!("GNAI request body cannot be cloned"))?
                    .build()?;
                // Pass the effective session setting to the wire adapter without sending it upstream.
                let effort = built.headers_mut().remove(EFFORT_HEADER);
                let Some(bytes) = built.body().and_then(reqwest::Body::as_bytes) else {
                    return Ok(request);
                };
                let mut payload: Value = serde_json::from_slice(bytes)?;
                if let Some(deployment) = wire_catalog.iter().find(|m| {
                    payload["model"] == m.name
                        || m.aliases.iter().any(|alias| payload["model"] == *alias)
                }) {
                    deployment.apply_wire_controls(
                        &mut payload,
                        effort.as_ref().and_then(|value| value.to_str().ok()),
                    );
                }
                let (client, _) = request.build_split();
                Ok(reqwest::RequestBuilder::from_parts(client, built).json(&payload))
            },
        );
        Ok(Self {
            catalog,
            metadata_client,
            chat: OpenAiCompatibleProvider::new(
                "gnai".into(),
                chat_client.with_request_builder(controls.clone()),
                "providers/openai/v1/".into(),
            ),
            responses: OpenAiProviderBuilder::new(
                responses_client.with_request_builder(controls.clone()),
            )
            .name("gnai")
            .base_path("providers/openai/v1/responses")
            .skip_canonical_filtering(true)
            .build(),
            messages: AnthropicProviderBuilder::new(
                messages_client
                    .with_request_builder(controls)
                    .with_header("anthropic-version", ANTHROPIC_API_VERSION)?,
            )
            .name("gnai")
            .skip_canonical_filtering(true)
            .build(),
        })
    }

    fn deployment(&self, name: &str) -> Result<&Deployment, ProviderError> {
        self.catalog
            .iter()
            .find(|m| m.name == name || m.aliases.iter().any(|a| a == name))
            .ok_or_else(|| {
                ProviderError::InvalidValue(format!("GNAI model {name:?} is unavailable"))
            })
    }
}

#[async_trait]
impl Provider for GnaiProvider {
    fn get_name(&self) -> &str {
        "gnai"
    }
    fn skip_canonical_filtering(&self) -> bool {
        true
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        self.metadata_client
            .refresh_credentials()
            .await
            .map_err(|e| ProviderError::Authentication(e.to_string()))
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(self.catalog.iter().map(|m| m.name.clone()).collect())
    }

    async fn fetch_supported_model_info(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(self.catalog.iter().map(Deployment::info).collect())
    }

    async fn fetch_model_info(&self, model_name: &str) -> Result<ModelInfo, ProviderError> {
        Ok(self.deployment(model_name)?.info())
    }

    async fn get_context_limit(&self, model: &str, override_limit: Option<usize>) -> usize {
        crate::context_limit::ContextLimitResolver::new("gnai")
            .resolve(model, override_limit, || async {
                self.fetch_model_info(model)
                    .await
                    .map(|info| info.context_limit)
            })
            .await
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let deployment = self.deployment(&model_config.model_name)?;
        let config = deployment.config(model_config)?;
        match deployment.route()? {
            Route::Messages => {
                self.messages
                    .stream_for_model(&config, &config.model_name, system, messages, tools)
                    .await
            }
            Route::Responses => {
                self.responses
                    .stream_for_model(
                        &config,
                        &config.model_name,
                        &deployment.name,
                        system,
                        messages,
                        tools,
                    )
                    .await
            }
            Route::Completions => {
                self.chat
                    .stream_for_model(
                        &config,
                        &config.model_name,
                        &deployment.name,
                        system,
                        messages,
                        tools,
                    )
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_client::AuthMethod;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn native_routes_preserve_alias_auth_attribution_and_terminal_failures() {
        let server = MockServer::start().await;
        let deployments = json!([
            {"name":"messages-deployment", "aliases":["MessagesAlias"], "capabilities":["chat","anthropic-messages"],
             "extra":{"adaptive_thinking":true,"effort":"low"}},
            {"name":"responses-deployment", "aliases":["ResponsesAlias"], "capabilities":["chat","openai-responses"]},
            {"name":"completions-deployment", "aliases":["CompletionsAlias"], "capabilities":["chat","openai-completions"]}
        ]);
        Mock::given(method("GET"))
            .and(path("/meta"))
            .and(header("authorization", "Bearer secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models":deployments})))
            .expect(1)
            .mount(&server)
            .await;
        for endpoint in [
            "/providers/anthropic/v1/messages",
            "/providers/openai/v1/responses",
            "/providers/openai/v1/chat/completions",
        ] {
            Mock::given(method("POST"))
                .and(path(endpoint))
                .and(header("authorization", "Bearer secret"))
                .and(header("x-gnai-context", "ask-acceptance"))
                .respond_with(ResponseTemplate::new(429).set_body_json(
                    json!({"error":{"message":"quota exhausted","type":"rate_limit_error"}}),
                ))
                .expect(1..)
                .mount(&server)
                .await;
        }
        let client = |url| {
            ApiClient::new_with_tls(url, AuthMethod::BearerToken("secret".into()), None)
                .unwrap()
                .with_header("X-GNAI-Context", "ask-acceptance")
                .unwrap()
        };
        let provider = GnaiProvider::create(
            client(server.uri()),
            client(server.uri()),
            client(server.uri()),
            client(format!("{}/providers/anthropic", server.uri())),
        )
        .await
        .unwrap();
        for alias in ["MessagesAlias", "ResponsesAlias", "CompletionsAlias"] {
            let result = provider
                .stream(
                    &ModelConfig::new(alias),
                    "system",
                    &[Message::user().with_text("question")],
                    &[],
                )
                .await;
            assert!(
                matches!(result, Err(ProviderError::RateLimitExceeded { .. })),
                "{alias} should fail structurally"
            );
        }
        let requests = server.received_requests().await.unwrap();
        for request in requests.iter().filter(|r| r.method == "POST") {
            let alias = match request.url.path() {
                "/providers/anthropic/v1/messages" => "MessagesAlias",
                "/providers/openai/v1/responses" => "ResponsesAlias",
                "/providers/openai/v1/chat/completions" => "CompletionsAlias",
                other => panic!("unexpected route {other}"),
            };
            assert_eq!(request.body_json::<Value>().unwrap()["model"], alias);
            assert!(!request.headers.contains_key(EFFORT_HEADER));
        }
        let messages = requests
            .iter()
            .find(|r| r.url.path().ends_with("/messages"))
            .unwrap()
            .body_json::<Value>()
            .unwrap();
        assert_eq!(messages["thinking"]["type"], "adaptive");
        assert_eq!(messages["output_config"]["effort"], "low");
    }

    #[test]
    fn routes_and_adapts_existing_metadata() {
        let deployment: Deployment = serde_json::from_value(json!({
            "name": "deployment", "aliases": ["ExactAlias"],
            "capabilities": ["chat", "openai-completions", "openai-responses", "anthropic-messages"],
            "extra": {"adaptive_thinking": true, "effort": "low"}
        })).unwrap();
        assert_eq!(deployment.route().unwrap(), Route::Messages);
        let config = deployment.config(&ModelConfig::new("ExactAlias")).unwrap();
        assert_eq!(config.model_name, "ExactAlias");
        let mut payload = json!({"model":"ExactAlias", "messages":[]});
        deployment.apply_wire_controls(&mut payload, Some("low"));
        assert_eq!(payload["thinking"]["type"], "adaptive");
        assert_eq!(payload["output_config"]["effort"], "low");
        assert!(deployment.info().context_limit.is_none());
    }
}
