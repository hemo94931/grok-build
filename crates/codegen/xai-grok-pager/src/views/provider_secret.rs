//! Fixed-mask renderer for dedicated provider API-key entry.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use crate::app::provider_auth::ProviderSecretState;
use crate::theme::Theme;

pub(crate) const PROVIDER_SECRET_HEIGHT: u16 = 6;
const FIXED_MASK: &str = "••••••••";

pub(crate) fn provider_secret_height(available_height: u16) -> u16 {
    PROVIDER_SECRET_HEIGHT.min(available_height)
}

pub(crate) fn render_provider_secret(
    buf: &mut Buffer,
    area: Rect,
    state: &ProviderSecretState,
    theme: &Theme,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let bg = theme.bg_visual;
    buf.set_style(area, Style::default().bg(bg));
    for y in area.y..area.y.saturating_add(area.height) {
        if let Some(cell) = buf.cell_mut((area.x, y)) {
            cell.set_symbol(crate::glyphs::accent_bar());
            cell.set_style(Style::default().fg(theme.accent_user).bg(bg));
        }
    }

    let x = area.x.saturating_add(3);
    let width = area.width.saturating_sub(5) as usize;
    if width == 0 {
        return;
    }
    let safe_provider =
        crate::views::session_title::sanitize_display_text(&state.request.provider_display_name);
    let safe_prompt = crate::views::session_title::sanitize_display_text(&state.request.prompt);
    let title = format!("{} API key", safe_provider.trim());
    if area.height > 0 {
        buf.set_stringn(
            x,
            area.y,
            title,
            width,
            Style::default()
                .fg(theme.text_primary)
                .bg(bg)
                .add_modifier(Modifier::BOLD),
        );
    }
    if area.height > 1 {
        buf.set_stringn(
            x,
            area.y + 1,
            safe_prompt.as_ref(),
            width,
            Style::default().fg(theme.gray).bg(bg),
        );
    }
    if area.height > 2 {
        let value = if state.has_value() {
            FIXED_MASK
        } else {
            "API key stays hidden"
        };
        buf.set_stringn(
            x,
            area.y + 2,
            value,
            width,
            Style::default()
                .fg(if state.has_value() {
                    theme.text_primary
                } else {
                    theme.gray
                })
                .bg(theme.bg_base)
                .add_modifier(Modifier::BOLD),
        );
    }
    if area.height > 3 && state.blank_error() {
        buf.set_stringn(
            x,
            area.y + 3,
            "API key cannot be empty",
            width,
            Style::default().fg(theme.accent_error).bg(bg),
        );
    }
    if area.height > 4 {
        buf.set_stringn(
            x,
            area.y + area.height - 1,
            "Enter save  ·  Esc cancel",
            width,
            Style::default().fg(theme.gray).bg(bg),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::Event;
    use xai_acp_lib::PromptSecretRequest;

    fn state() -> ProviderSecretState {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        ProviderSecretState::new(
            PromptSecretRequest {
                provider: "anthropic".into(),
                provider_display_name: "Anthropic".into(),
                session_id: "s1".into(),
                prompt: "Enter your API key".into(),
                request_seq: None,
            },
            tx,
        )
    }

    fn rendered(mut state: ProviderSecretState, input: &str) -> String {
        state.handle_event(&Event::Paste(input.to_owned()));
        let area = Rect::new(0, 0, 80, PROVIDER_SECRET_HEIGHT);
        let mut buf = Buffer::empty(area);
        render_provider_secret(&mut buf, area, &state, &Theme::current());
        buf.content().iter().map(|cell| cell.symbol()).collect()
    }

    #[test]
    fn rendered_secret_is_constant_mask_and_never_contains_plaintext() {
        let sentinel = "sk-screenshot-secret-sentinel";
        let short = rendered(state(), "x");
        let long = rendered(state(), sentinel);
        assert!(short.contains(FIXED_MASK));
        assert!(long.contains(FIXED_MASK));
        assert!(!long.contains(sentinel));
        assert!(!long.contains("screenshot-secret"));
        assert_eq!(short.matches(FIXED_MASK).count(), 1);
        assert_eq!(long.matches(FIXED_MASK).count(), 1);
    }
}
