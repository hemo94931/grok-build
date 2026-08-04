//! `/login` -- log in or re-authenticate with your account.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};

use super::providers::{normalize_provider, provider_items};

pub struct LoginCommand;

impl SlashCommand for LoginCommand {
    fn name(&self) -> &str {
        "login"
    }

    fn description(&self) -> &str {
        "Log in or re-authenticate with your account"
    }

    fn usage(&self) -> &str {
        "/login <provider>"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn args_required(&self) -> bool {
        true
    }

    fn arg_placeholder(&self) -> Option<&str> {
        Some("<provider>")
    }

    fn suggest_args(&self, _ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        Some(provider_items())
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        match normalize_provider(args) {
            Some("xai") => CommandResult::Action(Action::Login),
            Some(provider) => CommandResult::Action(Action::ProviderLogin(provider.to_owned())),
            None => CommandResult::Error(format!("Unknown provider: {}", args.trim())),
        }
    }
}
