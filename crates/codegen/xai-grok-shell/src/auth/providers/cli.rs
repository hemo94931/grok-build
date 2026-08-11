use std::io::{self, IsTerminal, Write};

use anyhow::{Context, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::{
    AuthInteraction, AuthNotification, AuthPrompt, AuthPromptResponse, AuthSecret,
    ProviderCredentialMethod, ProviderId, ProviderLoginOption, SelectOption, provider_descriptor,
    provider_login_options,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderTarget {
    Xai,
    Provider(ProviderId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProviderLoginTarget {
    Xai,
    Provider {
        provider: ProviderId,
        method: ProviderCredentialMethod,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CliLoginOption {
    Xai,
    Provider(ProviderLoginOption),
}

pub(crate) fn cli_login_options() -> Vec<CliLoginOption> {
    std::iter::once(CliLoginOption::Xai)
        .chain(
            provider_login_options()
                .into_iter()
                .map(CliLoginOption::Provider),
        )
        .collect()
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
        let selected = parse_selection(&value)?;
        match selected {
            1 => Ok(ProviderTarget::Xai),
            value if (2..=ProviderId::ALL.len() + 1).contains(&value) => {
                Ok(ProviderTarget::Provider(ProviderId::ALL[value - 2]))
            }
            _ => bail!(
                "selection must be between 1 and {}",
                ProviderId::ALL.len() + 1
            ),
        }
    })
    .await
    .context("provider selection task failed")?
}

pub(crate) async fn select_login_target() -> anyhow::Result<ProviderLoginTarget> {
    tokio::task::spawn_blocking(move || {
        let options = cli_login_options();
        eprintln!("Choose a provider login method:");
        for (index, option) in options.iter().enumerate() {
            match option {
                CliLoginOption::Xai => eprintln!("  {}. xAI", index + 1),
                CliLoginOption::Provider(option) => {
                    eprintln!(
                        "  {}. {} ({})",
                        index + 1,
                        option.display_name,
                        option.method.display_name()
                    );
                }
            }
        }
        eprint!("Selection [1]: ");
        io::stderr().flush()?;
        let value = read_line()?;
        let selected = parse_selection(&value)?;
        match options.get(selected.saturating_sub(1)) {
            Some(CliLoginOption::Xai) => Ok(ProviderLoginTarget::Xai),
            Some(CliLoginOption::Provider(option)) => Ok(ProviderLoginTarget::Provider {
                provider: option.provider,
                method: option.method,
            }),
            None => bail!("selection must be between 1 and {}", options.len()),
        }
    })
    .await
    .context("provider login selection task failed")?
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

    async fn prompt(&self, prompt: AuthPrompt) -> anyhow::Result<AuthPromptResponse> {
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
                read_line().map(AuthPromptResponse::Text)
            }
            AuthPrompt::Select { message, options } => {
                select_option(&message, &options).map(AuthPromptResponse::Text)
            }
            AuthPrompt::Secret {
                message,
                placeholder,
            } => {
                eprintln!("{message}");
                if !placeholder.is_empty() {
                    eprint!("{placeholder}: ");
                }
                io::stderr().flush()?;
                read_secret_line()
                    .and_then(normalize_secret_input)
                    .map(AuthSecret::new)
                    .map(AuthPromptResponse::Secret)
            }
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
    let selected = parse_selection(&value)?;
    options
        .get(selected.saturating_sub(1))
        .map(|option| option.id.clone())
        .context("selection is out of range")
}

fn parse_selection(value: &str) -> anyhow::Result<usize> {
    if value.trim().is_empty() {
        Ok(1)
    } else {
        value
            .trim()
            .parse::<usize>()
            .context("selection must be a number")
    }
}

fn read_line() -> anyhow::Result<String> {
    let mut value = String::new();
    if io::stdin().read_line(&mut value)? == 0 {
        bail!("standard input closed during authentication");
    }
    Ok(value.trim_end_matches(['\r', '\n']).to_owned())
}

fn normalize_secret_input(value: String) -> anyhow::Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("API key cannot be empty");
    }
    Ok(value.to_owned())
}

fn read_secret_line() -> anyhow::Result<String> {
    if !io::stdin().is_terminal() {
        return read_line();
    }
    read_secret_line_no_echo()
}

#[cfg(unix)]
struct TerminalEchoGuard {
    fd: libc::c_int,
    original: libc::termios,
    active: bool,
}

#[cfg(unix)]
impl TerminalEchoGuard {
    fn disable(fd: libc::c_int) -> anyhow::Result<Self> {
        use std::mem::MaybeUninit;

        let mut original = MaybeUninit::<libc::termios>::uninit();
        // SAFETY: `original` points to valid, writable storage for tcgetattr.
        if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error()).context("failed to read terminal mode");
        }
        // SAFETY: tcgetattr succeeded, so `original` is initialized.
        let original = unsafe { original.assume_init() };
        let mut hidden = original;
        hidden.c_lflag &= !libc::ECHO;
        // SAFETY: `hidden` is a valid termios captured from this terminal.
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &hidden) } != 0 {
            return Err(std::io::Error::last_os_error()).context("failed to disable terminal echo");
        }
        Ok(Self {
            fd,
            original,
            active: true,
        })
    }

    fn restore(&mut self) -> anyhow::Result<()> {
        if !self.active {
            return Ok(());
        }
        // SAFETY: `original` was captured from this same live terminal fd.
        if unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) } != 0 {
            return Err(std::io::Error::last_os_error()).context("failed to restore terminal echo");
        }
        self.active = false;
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: best-effort panic/error fallback using the saved mode for
            // this same terminal fd. The explicit restore path reports errors.
            let _ = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) };
        }
    }
}

#[cfg(unix)]
fn read_secret_line_no_echo() -> anyhow::Result<String> {
    let mut echo = TerminalEchoGuard::disable(libc::STDIN_FILENO)?;
    let result = read_line();
    let restore = echo.restore();
    eprintln!();
    restore?;
    result
}

#[cfg(not(unix))]
fn read_secret_line_no_echo() -> anyhow::Result<String> {
    read_line()
}

pub(crate) fn report_provider_login(provider: ProviderId) {
    eprintln!(
        "Signed in to {}.",
        provider_descriptor(provider).display_name
    );
}

pub(crate) fn report_provider_api_key_login(provider: ProviderId) {
    eprintln!(
        "API key saved for {}.",
        provider_descriptor(provider).display_name
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_normalization_trims_and_rejects_blank_input() {
        assert_eq!(
            normalize_secret_input("  sk-secret  \n".to_owned()).unwrap(),
            "sk-secret"
        );
        assert!(normalize_secret_input(" \t\r\n ".to_owned()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn terminal_echo_guard_disables_and_restores_echo() {
        use std::os::fd::AsRawFd;

        use nix::pty::openpty;
        use nix::sys::termios::{LocalFlags, tcgetattr};

        let pty = openpty(None, None).unwrap();
        let fd = pty.slave.as_raw_fd();
        let before = tcgetattr(&pty.slave).unwrap();
        assert!(before.local_flags.contains(LocalFlags::ECHO));

        let mut guard = TerminalEchoGuard::disable(fd).unwrap();
        let hidden = tcgetattr(&pty.slave).unwrap();
        assert!(!hidden.local_flags.contains(LocalFlags::ECHO));
        guard.restore().unwrap();

        let restored = tcgetattr(&pty.slave).unwrap();
        assert!(restored.local_flags.contains(LocalFlags::ECHO));
    }
}
