use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::future::BoxFuture;
use goose_providers::api_client::{ApiClient, AuthMethod, AuthProvider, TlsConfig};
use goose_providers::base::{ProviderDescriptor, ProviderMetadata};
use goose_providers::gnai::GnaiProvider;

use crate::config::{Config, ExtensionConfig};
use crate::providers::base::ProviderDef;

struct GnaiAuth {
    helper: ApiClient,
}

#[async_trait]
impl AuthProvider for GnaiAuth {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let response = self.helper.response_get("").await?.error_for_status()?;
        let payload: serde_json::Value = response.json().await?;
        let token = payload["access_token"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("GNAI credential helper returned no token"))?;
        Ok(("Authorization".into(), format!("Bearer {token}")))
    }

    async fn refresh_credentials(&self) -> Result<()> {
        Ok(())
    }
}

pub struct GnaiProviderDef;

impl ProviderDescriptor for GnaiProviderDef {
    fn metadata() -> ProviderMetadata {
        GnaiProvider::metadata()
    }
}

impl ProviderDef for GnaiProviderDef {
    type Provider = GnaiProvider;

    fn from_env(
        _extensions: Vec<ExtensionConfig>,
        tls: Option<TlsConfig>,
    ) -> BoxFuture<'static, Result<GnaiProvider>> {
        Box::pin(async move {
            let config = Config::global();
            let root: String = config.get_param("GNAI_URL")?;
            let context: String = config
                .get_param("GNAI_CONTEXT")
                .unwrap_or_else(|_| "goose".into());
            let auth = || -> Result<AuthMethod> {
                if let Some(helper) = config
                    .get_param::<String>("GNAI_CREDENTIAL_URL")
                    .ok()
                    .filter(|value| !value.is_empty())
                {
                    let key = config.get_secret::<String>("GNAI_CREDENTIAL_KEY")?;
                    let helper =
                        ApiClient::new_with_tls(helper, AuthMethod::BearerToken(key), tls.clone())?
                            .with_loopback_http_only()?
                            .with_same_origin_redirects()?;
                    Ok(AuthMethod::Custom(Box::new(GnaiAuth { helper })))
                } else {
                    Ok(AuthMethod::BearerToken(config.get_secret("GNAI_TOKEN")?))
                }
            };
            let client = |url: String| -> Result<ApiClient> {
                ApiClient::new_with_tls(url, auth()?, tls.clone())?
                    .with_same_origin_redirects()?
                    .with_header("X-GNAI-Context", &context)
            };
            GnaiProvider::create(
                client(root.clone())?,
                client(root.clone())?,
                client(root.clone())?,
                client(format!(
                    "{}/providers/anthropic",
                    root.trim_end_matches('/')
                ))?,
            )
            .await
        })
    }
}
