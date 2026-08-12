use tokio_util::sync::CancellationToken;

use super::{AuthInteraction, LoginMode, ProviderCredential, ProviderRefreshOutcome};

pub(super) async fn login(
    _interaction: &dyn AuthInteraction,
    _mode: Option<LoginMode>,
) -> anyhow::Result<ProviderCredential> {
    anyhow::bail!("Z.AI providers support API-key login only")
}

pub(super) async fn refresh(
    _credential: ProviderCredential,
    _signal: CancellationToken,
) -> anyhow::Result<ProviderRefreshOutcome> {
    anyhow::bail!("Z.AI API keys cannot be refreshed")
}
