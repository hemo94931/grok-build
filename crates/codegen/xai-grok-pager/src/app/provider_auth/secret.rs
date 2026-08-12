//! Dedicated, non-recording provider secret input state.
//!
//! This state never touches the ordinary prompt widget, slash pipeline,
//! question annotations, actions, task results, scrollback, or input recorder.

use std::fmt;

use agent_client_protocol as acp;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use xai_acp_lib::{AcpResult, PromptSecretRequest, PromptSecretResponse, RedactedSecretBuffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SecretInputOutcome {
    Changed,
    Submitted,
    Cancelled,
}

pub(crate) struct ProviderSecretState {
    pub(crate) request: PromptSecretRequest,
    buffer: RedactedSecretBuffer,
    response_tx: Option<tokio::sync::oneshot::Sender<AcpResult<acp::ExtResponse>>>,
    blank_error: bool,
}

impl ProviderSecretState {
    pub(crate) fn new(
        request: PromptSecretRequest,
        response_tx: tokio::sync::oneshot::Sender<AcpResult<acp::ExtResponse>>,
    ) -> Self {
        Self {
            request,
            buffer: RedactedSecretBuffer::default(),
            response_tx: Some(response_tx),
            blank_error: false,
        }
    }

    pub(crate) fn has_value(&self) -> bool {
        !self.buffer.is_empty()
    }

    pub(crate) fn blank_error(&self) -> bool {
        self.blank_error
    }

    pub(crate) fn handle_event(&mut self, event: &Event) -> SecretInputOutcome {
        match event {
            Event::Paste(value) => {
                for ch in value.chars().filter(|ch| !ch.is_control()) {
                    self.buffer.push(ch);
                }
                self.blank_error = false;
                SecretInputOutcome::Changed
            }
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                if key.code == KeyCode::Esc
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    self.cancel();
                    return SecretInputOutcome::Cancelled;
                }
                match key.code {
                    KeyCode::Enter => self.submit(),
                    KeyCode::Backspace | KeyCode::Delete => {
                        self.buffer.pop();
                        self.blank_error = false;
                        SecretInputOutcome::Changed
                    }
                    KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        self.buffer.clear();
                        self.blank_error = false;
                        SecretInputOutcome::Changed
                    }
                    KeyCode::Char(ch)
                        if !key.modifiers.intersects(
                            KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                        ) && !ch.is_control() =>
                    {
                        self.buffer.push(ch);
                        self.blank_error = false;
                        SecretInputOutcome::Changed
                    }
                    _ => SecretInputOutcome::Changed,
                }
            }
            _ => SecretInputOutcome::Changed,
        }
    }

    fn submit(&mut self) -> SecretInputOutcome {
        let Some(secret) = self.buffer.take_trimmed() else {
            self.blank_error = true;
            return SecretInputOutcome::Changed;
        };
        if let Some(tx) = self.response_tx.take() {
            let response = PromptSecretResponse::Accepted { secret };
            let raw = serde_json::value::to_raw_value(&response)
                .expect("provider secret response serialization should not fail");
            tx.send(Ok(acp::ExtResponse::new(raw.into()))).ok();
        }
        SecretInputOutcome::Submitted
    }

    pub(crate) fn cancel(&mut self) {
        self.buffer.clear();
        if let Some(tx) = self.response_tx.take() {
            let raw = serde_json::value::to_raw_value(&PromptSecretResponse::Cancelled)
                .expect("provider secret cancellation serialization should not fail");
            tx.send(Ok(acp::ExtResponse::new(raw.into()))).ok();
        }
    }
}

impl fmt::Debug for ProviderSecretState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderSecretState")
            .field("provider", &self.request.provider)
            .field("session_id", &self.request.session_id)
            .field("request_seq", &self.request.request_seq)
            .field("buffer", &"[REDACTED]")
            .field("blank_error", &self.blank_error)
            .finish()
    }
}

impl Drop for ProviderSecretState {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventState};

    const SENTINEL: &str = "sk-provider-secret-sentinel";

    fn request() -> PromptSecretRequest {
        PromptSecretRequest {
            provider: "anthropic".into(),
            provider_display_name: "Anthropic".into(),
            session_id: "session-1".into(),
            prompt: "Enter Anthropic API key".into(),
            request_seq: Some(4),
        }
    }

    fn key(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    #[test]
    fn typing_paste_and_backspace_never_expose_secret_in_debug() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut state = ProviderSecretState::new(request(), tx);
        for ch in "sk-".chars() {
            assert_eq!(
                state.handle_event(&key(KeyCode::Char(ch), KeyModifiers::NONE)),
                SecretInputOutcome::Changed
            );
        }
        state.handle_event(&Event::Paste("provider-secret-sentinex\n".into()));
        state.handle_event(&key(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(state.has_value());
        let debug = format!("{state:?}");
        assert!(!debug.contains(SENTINEL));
        assert!(!debug.contains("sentinel"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn submit_uses_dedicated_response_without_annotations() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut state = ProviderSecretState::new(request(), tx);
        state.handle_event(&Event::Paste(format!("  {SENTINEL}\t\0\n")));
        assert_eq!(
            state.handle_event(&key(KeyCode::Enter, KeyModifiers::NONE)),
            SecretInputOutcome::Submitted
        );
        let response = rx.blocking_recv().unwrap().unwrap();
        let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
        assert_eq!(json["outcome"], "accepted");
        assert_eq!(json["secret"], SENTINEL);
        assert!(json.get("answers").is_none());
        assert!(json.get("annotations").is_none());
    }

    #[test]
    fn blank_submit_stays_open_and_cancel_replies_cancelled() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut state = ProviderSecretState::new(request(), tx);
        state.handle_event(&Event::Paste(" \t\n".into()));
        assert_eq!(
            state.handle_event(&key(KeyCode::Enter, KeyModifiers::NONE)),
            SecretInputOutcome::Changed
        );
        assert!(state.blank_error());
        assert_eq!(
            state.handle_event(&key(KeyCode::Esc, KeyModifiers::NONE)),
            SecretInputOutcome::Cancelled
        );
        let response = rx.blocking_recv().unwrap().unwrap();
        let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
        assert_eq!(json["outcome"], "cancelled");
        assert!(json.get("secret").is_none());
    }

    #[test]
    fn dropping_state_replies_cancelled() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        drop(ProviderSecretState::new(request(), tx));
        let response = rx.blocking_recv().unwrap().unwrap();
        let json: serde_json::Value = serde_json::from_str(response.0.get()).unwrap();
        assert_eq!(json["outcome"], "cancelled");
    }
}
