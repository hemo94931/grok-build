use super::*;

/// Boxed future returned by [`ModelsEndpoint::fetch_models`].
pub(crate) type ModelsFetchFuture =
    Pin<Box<dyn Future<Output = Option<IndexMap<String, ModelEntry>>> + Send>>;

/// Injectable `/v1/models` transport; tests inject a fake.
pub(crate) trait ModelsEndpoint: Send + Sync {
    fn fetch_models(
        &self,
        endpoints: config::EndpointsConfig,
        auth: Option<GrokAuth>,
        fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture;
}

/// Default transport: the real `/v1/models` fetch.
pub(crate) struct HttpModelsEndpoint;

pub(crate) type RadiusCatalogFetchFuture =
    Pin<Box<dyn Future<Output = anyhow::Result<Option<RadiusCatalogFetchResult>>> + Send>>;

#[derive(Clone, Debug)]
pub(crate) struct RadiusCatalogFetchResult {
    /// Radius gateway origin. Contains no credential material.
    pub(crate) origin: String,
    pub(crate) catalog: crate::auth::providers::RadiusGatewayConfig,
}

/// Injectable Radius `/v1/config` transport; tests inject fakes or a store-bound HTTP endpoint.
pub(crate) trait RadiusCatalogEndpoint: Send + Sync {
    fn fetch_radius_catalog(&self) -> RadiusCatalogFetchFuture;
}

pub(crate) struct HttpRadiusCatalogEndpoint {
    store: Option<crate::auth::providers::ProviderStore>,
}

impl HttpRadiusCatalogEndpoint {
    pub(crate) fn new() -> Self {
        Self { store: None }
    }

    #[cfg(test)]
    pub(crate) fn with_store(store: crate::auth::providers::ProviderStore) -> Self {
        Self { store: Some(store) }
    }
}

impl RadiusCatalogEndpoint for HttpRadiusCatalogEndpoint {
    fn fetch_radius_catalog(&self) -> RadiusCatalogFetchFuture {
        let store = self.store.clone();
        Box::pin(async move {
            use crate::auth::providers::{
                ProviderId, ProviderSecret, ProviderSlotState, ProviderStoredCredential,
            };
            let slot = match store {
                Some(store) => store.get(ProviderId::Radius)?,
                None => crate::auth::providers::fresh_stored_slot(ProviderId::Radius, None).await?,
            };
            let secret = match slot {
                ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => {
                    ProviderSecret::from_oauth(ProviderId::Radius, &credential)
                }
                ProviderSlotState::Known(ProviderStoredCredential::ApiKey(credential)) => {
                    ProviderSecret::from_api_key(ProviderId::Radius, credential.key)?
                }
                ProviderSlotState::Missing => {
                    match ProviderSecret::from_environment(ProviderId::Radius)? {
                        Some(secret) => secret,
                        None => return Ok(None),
                    }
                }
                ProviderSlotState::PresentUnsupportedOrInvalid => {
                    anyhow::bail!("Radius credential slot is unsupported or invalid")
                }
            };
            let origin = crate::auth::providers::gateway_cache_origin()?;
            let signal = tokio_util::sync::CancellationToken::new();
            let catalog =
                crate::auth::providers::load_gateway_config_for_catalog(&secret.token, &signal)
                    .await?;
            Ok(Some(RadiusCatalogFetchResult { origin, catalog }))
        })
    }
}

impl ModelsEndpoint for HttpModelsEndpoint {
    fn fetch_models(
        &self,
        endpoints: config::EndpointsConfig,
        auth: Option<GrokAuth>,
        fetch_auth: ModelFetchAuth,
    ) -> ModelsFetchFuture {
        Box::pin(fetch_models_async(endpoints, auth, fetch_auth))
    }
}

pub(crate) async fn fetch_models_async(
    endpoints: config::EndpointsConfig,
    auth: Option<GrokAuth>,
    fetch_auth: ModelFetchAuth,
) -> Option<IndexMap<String, ModelEntry>> {
    tokio::task::spawn_blocking(move || {
        prefetch_models_blocking(&endpoints, auth.as_ref(), fetch_auth)
    })
    .await
    .unwrap_or(None)
}
