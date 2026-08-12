//! `/logout` -- remove auth credentials and return to the login screen.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};

use super::providers::{normalize_known_provider, provider_items};

pub struct LogoutCommand;

impl SlashCommand for LogoutCommand {
    fn name(&self) -> &str {
        "logout"
    }

    fn description(&self) -> &str {
        "Log out and return to the login screen"
    }

    fn usage(&self) -> &str {
        "/logout <provider>"
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
        match normalize_known_provider(args) {
            Some(provider) if provider == "xai" => CommandResult::Action(Action::Logout),
            Some(provider) => CommandResult::Action(Action::ProviderLogout(provider)),
            None => CommandResult::Error(format!("Unknown provider: {}", args.trim())),
        }
    }
}
