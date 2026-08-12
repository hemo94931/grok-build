//! Pager-owned provider-auth state and protocol projections.
//!
//! Provider capability/method facts come from `x.ai/providerAuth/info`; this
//! module only turns that metadata into typed login intents and UI rows.

use serde::Deserialize;
use xai_acp_lib::{ProviderAuthMethod, ProviderAuthRemedy};

use crate::slash::command::ArgItem;

pub(crate) mod secret;

pub(crate) use secret::{ProviderSecretState, SecretInputOutcome};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderLoginIntent {
    Menu,
    Provider {
        provider: String,
        method: Option<ProviderAuthMethod>,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderAuthInfo {
    pub(crate) id: String,
    pub(crate) display_name: String,
    pub(crate) supported_methods: Vec<ProviderAuthMethod>,
    #[serde(default)]
    pub(crate) authenticated: bool,
    #[serde(default)]
    pub(crate) credential_type: Option<ProviderAuthMethod>,
    #[serde(default)]
    pub(crate) model_count: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProviderAuthInfoResponse {
    pub(crate) providers: Vec<ProviderAuthInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedProviderLogin {
    pub(crate) provider: String,
    pub(crate) display_name: String,
    pub(crate) method: ProviderAuthMethod,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderLoginSuccess {
    pub(crate) provider: String,
    pub(crate) display_name: String,
    pub(crate) method: ProviderAuthMethod,
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) catalog_refreshed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingProviderLogin {
    pub(crate) provider: String,
    pub(crate) display_name: String,
    pub(crate) method: ProviderAuthMethod,
    pub(crate) request_seq: u64,
    pub(crate) owns_input: bool,
    pub(crate) cancelled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingProviderReauth {
    pub(crate) prompt_id: Option<String>,
    pub(crate) remedy: ProviderAuthRemedy,
}

pub(crate) fn resolve_provider_login(
    providers: &[ProviderAuthInfo],
    provider: &str,
    requested_method: Option<ProviderAuthMethod>,
) -> Result<ResolvedProviderLogin, String> {
    let normalized = normalize_provider_id(provider);
    let Some(info) = providers.iter().find(|info| {
        info.id.eq_ignore_ascii_case(&normalized)
            || info.display_name.eq_ignore_ascii_case(provider.trim())
    }) else {
        return Err(format!("Unknown provider: {}", provider.trim()));
    };

    let method = requested_method.unwrap_or_else(|| {
        if info.supported_methods.contains(&ProviderAuthMethod::OAuth) {
            ProviderAuthMethod::OAuth
        } else {
            ProviderAuthMethod::ApiKey
        }
    });
    if !info.supported_methods.contains(&method) {
        return Err(format!(
            "{} does not support {} login",
            info.display_name,
            method.display_name()
        ));
    }
    Ok(ResolvedProviderLogin {
        provider: info.id.clone(),
        display_name: info.display_name.clone(),
        method,
    })
}

pub(crate) fn login_selector_items(providers: &[ProviderAuthInfo]) -> Vec<ArgItem> {
    let mut items = vec![ArgItem {
        display: "xAI — account login".to_owned(),
        match_text: "xAI x.ai xai account login OAuth".to_owned(),
        insert_text: "xai".to_owned(),
        description: "Grok and xAI models".to_owned(),
    }];
    for provider in providers {
        for method in &provider.supported_methods {
            let flag = match method {
                ProviderAuthMethod::OAuth => "--oauth",
                ProviderAuthMethod::ApiKey => "--api-key",
            };
            items.push(ArgItem {
                display: format!("{} — {}", provider.display_name, method.display_name()),
                match_text: format!(
                    "{} {} {} {}",
                    provider.display_name,
                    provider.id,
                    method.display_name(),
                    method.as_str()
                ),
                insert_text: format!("{} {flag}", provider.id),
                description: match method {
                    ProviderAuthMethod::OAuth => {
                        format!("Sign in to {} with OAuth", provider.display_name)
                    }
                    ProviderAuthMethod::ApiKey => {
                        format!("Store an API key for {}", provider.display_name)
                    }
                },
            });
        }
    }
    items
}

pub(crate) fn provider_completion_items(providers: &[ProviderAuthInfo]) -> Vec<ArgItem> {
    let mut items = vec![ArgItem {
        display: "xAI".to_owned(),
        match_text: "xAI x.ai xai".to_owned(),
        insert_text: "xai".to_owned(),
        description: "Grok and xAI models".to_owned(),
    }];
    items.extend(providers.iter().map(|provider| {
        ArgItem {
            display: provider.display_name.clone(),
            match_text: format!("{} {}", provider.display_name, provider.id),
            insert_text: provider.id.clone(),
            description: provider
                .supported_methods
                .iter()
                .map(|method| method.display_name())
                .collect::<Vec<_>>()
                .join(" or "),
        }
    }));
    items
}

fn normalize_provider_id(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

impl crate::app::agent_view::AgentView {
    pub(crate) fn cancel_provider_secret(&mut self) {
        if let Some(mut state) = self.provider_secret.take() {
            state.cancel();
        }
    }

    pub(crate) fn request_provider_login_cancel(&mut self) -> Option<(String, u64)> {
        let pending = self.pending_provider_login.as_mut()?;
        if pending.method != ProviderAuthMethod::ApiKey || pending.cancelled {
            return None;
        }
        pending.cancelled = true;
        Some((pending.provider.clone(), pending.request_seq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn providers() -> Vec<ProviderAuthInfo> {
        vec![
            ProviderAuthInfo {
                id: "anthropic".into(),
                display_name: "Anthropic".into(),
                supported_methods: vec![ProviderAuthMethod::OAuth, ProviderAuthMethod::ApiKey],
                authenticated: false,
                credential_type: None,
                model_count: 2,
            },
            ProviderAuthInfo {
                id: "openai-codex".into(),
                display_name: "OpenAI Codex".into(),
                supported_methods: vec![ProviderAuthMethod::OAuth],
                authenticated: false,
                credential_type: None,
                model_count: 3,
            },
            ProviderAuthInfo {
                id: "key-only".into(),
                display_name: "Key Only".into(),
                supported_methods: vec![ProviderAuthMethod::ApiKey],
                authenticated: false,
                credential_type: None,
                model_count: 1,
            },
        ]
    }

    #[test]
    fn provider_info_and_login_success_parse_shell_wire_method_names() {
        let info: ProviderAuthInfoResponse = serde_json::from_value(serde_json::json!({
            "providers": [{
                "id": "anthropic",
                "displayName": "Anthropic",
                "supportedMethods": ["oauth", "api_key"],
                "authenticated": false,
                "credentialType": null,
                "modelCount": 2
            }]
        }))
        .unwrap();
        assert_eq!(
            info.providers[0].supported_methods,
            vec![ProviderAuthMethod::OAuth, ProviderAuthMethod::ApiKey]
        );

        let success: ProviderLoginSuccess = serde_json::from_value(serde_json::json!({
            "provider": "anthropic",
            "displayName": "Anthropic",
            "method": "oauth",
            "message": "Signed in to Anthropic with OAuth.",
            "catalogRefreshed": null
        }))
        .unwrap();
        assert_eq!(success.method, ProviderAuthMethod::OAuth);
    }

    #[test]
    fn selector_flattens_provider_methods_and_includes_xai() {
        let items = login_selector_items(&providers());
        assert_eq!(items.len(), 5);
        assert_eq!(items[0].insert_text, "xai");
        assert!(
            items
                .iter()
                .any(|item| item.insert_text == "anthropic --oauth")
        );
        assert!(
            items
                .iter()
                .any(|item| item.insert_text == "anthropic --api-key")
        );
        assert!(
            items
                .iter()
                .any(|item| item.insert_text == "key-only --api-key")
        );
    }

    #[test]
    fn resolution_defaults_dual_to_oauth_and_key_only_to_api_key() {
        let providers = providers();
        assert_eq!(
            resolve_provider_login(&providers, "anthropic", None)
                .unwrap()
                .method,
            ProviderAuthMethod::OAuth
        );
        assert_eq!(
            resolve_provider_login(&providers, "Key Only", None)
                .unwrap()
                .method,
            ProviderAuthMethod::ApiKey
        );
        assert_eq!(
            resolve_provider_login(&providers, "anthropic", Some(ProviderAuthMethod::ApiKey))
                .unwrap()
                .method,
            ProviderAuthMethod::ApiKey
        );
    }

    #[test]
    fn resolution_rejects_unsupported_and_unknown_methods_locally() {
        let providers = providers();
        let codex =
            resolve_provider_login(&providers, "OpenAI Codex", Some(ProviderAuthMethod::ApiKey))
                .unwrap_err();
        assert!(codex.contains("does not support API key"));
        let key_only =
            resolve_provider_login(&providers, "key-only", Some(ProviderAuthMethod::OAuth))
                .unwrap_err();
        assert!(key_only.contains("does not support OAuth"));
        assert!(resolve_provider_login(&providers, "missing", None).is_err());
    }
}
