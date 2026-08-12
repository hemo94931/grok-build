//! `/login` -- log in or re-authenticate with an account/provider.

use crate::app::actions::Action;
use crate::app::provider_auth::ProviderLoginIntent;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};
use xai_acp_lib::ProviderAuthMethod;

use super::providers::{normalize_known_provider, provider_items};

pub struct LoginCommand;

impl SlashCommand for LoginCommand {
    fn name(&self) -> &str {
        "login"
    }

    fn description(&self) -> &str {
        "Log in or re-authenticate with your account"
    }

    fn usage(&self) -> &str {
        "/login [provider] [--oauth|--api-key]"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn args_required(&self) -> bool {
        false
    }

    fn arg_placeholder(&self) -> Option<&str> {
        Some("[provider] [--oauth|--api-key]")
    }

    fn suggest_args(&self, _ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        Some(provider_items())
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        match parse_login_args(args) {
            Ok(ProviderLoginIntent::Menu) => {
                CommandResult::Action(Action::ProviderLogin(ProviderLoginIntent::Menu))
            }
            Ok(ProviderLoginIntent::Provider { provider, method }) if provider == "xai" => {
                if method.is_some() {
                    CommandResult::Error(
                        "xAI login does not accept provider --oauth/--api-key flags".to_owned(),
                    )
                } else {
                    CommandResult::Action(Action::Login)
                }
            }
            Ok(intent) => CommandResult::Action(Action::ProviderLogin(intent)),
            Err(error) => CommandResult::Error(error),
        }
    }
}

fn parse_login_args(args: &str) -> Result<ProviderLoginIntent, String> {
    if args.trim().is_empty() {
        return Ok(ProviderLoginIntent::Menu);
    }

    let mut provider_parts = Vec::new();
    let mut method = None;
    for token in args.split_whitespace() {
        let parsed_method = match token {
            "--oauth" => Some(ProviderAuthMethod::OAuth),
            "--api-key" => Some(ProviderAuthMethod::ApiKey),
            value if value.starts_with("--") => {
                return Err(format!("Unknown /login argument: {value}"));
            }
            value => {
                provider_parts.push(value);
                None
            }
        };
        if let Some(parsed_method) = parsed_method {
            if let Some(existing) = method {
                if existing == parsed_method {
                    return Err(format!("Duplicate /login argument: {token}"));
                }
                return Err("Choose only one of --oauth or --api-key".to_owned());
            }
            method = Some(parsed_method);
        }
    }
    if provider_parts.is_empty() {
        return Err("A provider is required when selecting a login method".to_owned());
    }
    let provider_text = provider_parts.join(" ");
    let provider = normalize_known_provider(&provider_text)
        .ok_or_else(|| format!("Unknown provider: {provider_text}"))?;
    Ok(ProviderLoginIntent::Provider { provider, method })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_login_opens_menu_and_explicit_methods_are_typed() {
        assert_eq!(parse_login_args("").unwrap(), ProviderLoginIntent::Menu);
        assert_eq!(
            parse_login_args("anthropic --api-key").unwrap(),
            ProviderLoginIntent::Provider {
                provider: "anthropic".into(),
                method: Some(ProviderAuthMethod::ApiKey),
            }
        );
        assert_eq!(
            parse_login_args("OpenAI Codex --oauth").unwrap(),
            ProviderLoginIntent::Provider {
                provider: "openai-codex".into(),
                method: Some(ProviderAuthMethod::OAuth),
            }
        );
    }

    #[test]
    fn login_parser_rejects_duplicate_conflicting_and_unknown_flags() {
        assert!(parse_login_args("anthropic --api-key --api-key").is_err());
        assert!(parse_login_args("anthropic --api-key --oauth").is_err());
        assert!(parse_login_args("anthropic --wat").is_err());
        assert!(parse_login_args("--api-key").is_err());
    }
}
