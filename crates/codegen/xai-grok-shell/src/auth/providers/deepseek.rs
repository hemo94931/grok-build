use tokio_util::sync::CancellationToken;

use super::{AuthInteraction, LoginMode, ProviderCredential, ProviderRefreshOutcome};

pub(super) async fn login(
    _interaction: &dyn AuthInteraction,
    _mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    anyhow::bail!("DeepSeek supports API-key login only")
}

pub(super) async fn refresh(
    _credential: ProviderCredential,
    _signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    anyhow::bail!("DeepSeek API keys cannot be refreshed")
}
