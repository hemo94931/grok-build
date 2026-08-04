use std::io::{self, IsTerminal, Write};

use anyhow::{Context, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    AuthInteraction, AuthNotification, AuthPrompt, ProviderId, SelectOption, provider_descriptor,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderTarget {
    Xai,
    Provider(ProviderId),
}

impl ProviderTarget {
    pub(crate) fn parse(value: &str) -> anyhow::Result<Self> {
        if matches!(value.trim().to_ascii_lowercase().as_str(), "xai" | "x.ai") {
            Ok(Self::Xai)
        } else {
            value.parse().map(Self::Provider)
        }
    }
}

pub(crate) fn terminal_is_interactive() -> bool {
    io::stdin().is_terminal() && io::stderr().is_terminal()
}

pub(crate) async fn select_target(action: &str) -> anyhow::Result<ProviderTarget> {
    let action = action.to_owned();
    tokio::task::spawn_blocking(move || {
        eprintln!("Choose a provider to {action}:");
        eprintln!("  1. xAI");
        for (index, provider) in ProviderId::ALL.iter().enumerate() {
            eprintln!("  {}. {}", index + 2, provider.display_name());
        }
        eprint!("Selection [1]: ");
        io::stderr().flush()?;
        let value = read_line()?;
        let selected = if value.trim().is_empty() {
            1
        } else {
            value
                .trim()
                .parse::<usize>()
                .context("selection must be a number")?
        };
        match selected {
            1 => Ok(ProviderTarget::Xai),
            value @ 2..=7 => Ok(ProviderTarget::Provider(ProviderId::ALL[value - 2])),
            _ => bail!("selection must be between 1 and 7"),
        }
    })
    .await
    .context("provider selection task failed")?
}

#[derive(Debug, Default)]
pub(crate) struct CliAuthInteraction {
    signal: CancellationToken,
}

impl CliAuthInteraction {
    pub(crate) fn new() -> Self {
        Self {
            signal: CancellationToken::new(),
        }
    }
}

#[async_trait(?Send)]
impl AuthInteraction for CliAuthInteraction {
    fn signal(&self) -> CancellationToken {
        self.signal.clone()
    }

    async fn notify(&self, notification: AuthNotification) -> anyhow::Result<()> {
        match notification {
            AuthNotification::AuthUrl { url, instructions } => {
                eprintln!("{instructions}");
                eprintln!("{url}");
                if terminal_is_interactive() {
                    let _ = webbrowser::open(&url);
                }
            }
            AuthNotification::DeviceCode {
                user_code,
                verification_uri,
                ..
            } => {
                eprintln!("Open {verification_uri}");
                eprintln!("Enter code: {user_code}");
            }
            AuthNotification::Progress { message } | AuthNotification::Info { message, .. } => {
                eprintln!("{message}");
            }
        }
        Ok(())
    }

    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<String> {
        tokio::task::spawn_blocking(move || match prompt {
            AuthPrompt::Text {
                message,
                placeholder,
            }
            | AuthPrompt::ManualCode {
                message,
                placeholder,
            } => {
                eprintln!("{message}");
                if !placeholder.is_empty() {
                    eprint!("{placeholder}: ");
                }
                io::stderr().flush()?;
                read_line()
            }
            AuthPrompt::Select { message, options } => select_option(&message, &options),
        })
        .await
        .context("authentication prompt task failed")?
    }
}

fn select_option(message: &str, options: &[SelectOption]) -> anyhow::Result<String> {
    if options.is_empty() {
        bail!("authentication prompt has no choices");
    }
    eprintln!("{message}");
    for (index, option) in options.iter().enumerate() {
        eprintln!("  {}. {}", index + 1, option.label);
    }
    eprint!("Selection [1]: ");
    io::stderr().flush()?;
    let value = read_line()?;
    let selected = if value.trim().is_empty() {
        1
    } else {
        value
            .trim()
            .parse::<usize>()
            .context("selection must be a number")?
    };
    options
        .get(selected.saturating_sub(1))
        .map(|option| option.id.clone())
        .context("selection is out of range")
}

fn read_line() -> anyhow::Result<String> {
    let mut value = String::new();
    if io::stdin().read_line(&mut value)? == 0 {
        bail!("standard input closed during authentication");
    }
    Ok(value.trim_end_matches(['\r', '\n']).to_owned())
}

pub(crate) fn report_provider_login(provider: ProviderId) {
    eprintln!(
        "Signed in to {}.",
        provider_descriptor(provider).display_name
    );
}
