use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{ProviderId, ProviderRefreshOutcome};

pub(crate) const REFRESH_WINDOW: Duration = Duration::from_secs(5 * 60);
const PERMANENT_EXPIRY_MS: u64 = 9_007_199_254_740_991;

// ponytail: one process-wide mutex keeps the lock order boring; shard by store
// path only if provider login/refresh contention is ever measurable.
static PROCESS_MUTATION_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static TMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type RawCredentialTable = Map<String, Value>;

/// Canonical pi-compatible OAuth credential. Provider-specific fields are
/// retained in `metadata`, so newer providers can add fields without changing
/// the storage schema.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ProviderCredential {
    #[serde(rename = "type")]
    credential_type: String,
    pub(crate) access: String,
    #[serde(default)]
    pub(crate) refresh: String,
    pub(crate) expires: u64,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

impl ProviderCredential {
    pub(crate) fn oauth(
        access: impl Into<String>,
        refresh: impl Into<String>,
        expires: u64,
    ) -> Self {
        Self {
            credential_type: "oauth".to_owned(),
            access: access.into(),
            refresh: refresh.into(),
            expires,
            metadata: BTreeMap::new(),
        }
    }

    pub(crate) fn permanent(access: impl Into<String>) -> Self {
        Self::oauth(access, "", PERMANENT_EXPIRY_MS)
    }

    pub(crate) fn expires_within(&self, window: Duration) -> bool {
        self.expires <= unix_millis().saturating_add(window.as_millis() as u64)
    }

    pub(crate) fn metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub(crate) fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    pub(crate) fn metadata_strings(&self, key: &str) -> Option<Vec<String>> {
        self.metadata
            .get(key)?
            .as_array()?
            .iter()
            .map(|value| value.as_str().map(ToOwned::to_owned))
            .collect()
    }

    pub(crate) fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) {
        self.metadata.insert(key.into(), value.into());
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.credential_type != "oauth" {
            bail!("unsupported provider credential type");
        }
        if self.access.trim().is_empty() {
            bail!("provider credential is missing an access token");
        }
        Ok(())
    }
}

impl fmt::Debug for ProviderCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderCredential")
            .field("credential_type", &self.credential_type)
            .field("access", &"[REDACTED]")
            .field("refresh", &"[REDACTED]")
            .field("expires", &self.expires)
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Canonical stored API-key credential. Extra fields are retained so a future
/// writer can add provider metadata without an older writer dropping it when
/// the credential itself is intentionally re-saved.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ProviderApiKeyCredential {
    #[serde(rename = "type")]
    credential_type: String,
    pub(crate) key: String,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

impl ProviderApiKeyCredential {
    pub(crate) fn new(key: impl Into<String>) -> Self {
        Self {
            credential_type: "api_key".to_owned(),
            key: key.into(),
            metadata: BTreeMap::new(),
        }
    }

    pub(crate) fn metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub(crate) fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    pub(crate) fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) {
        self.metadata.insert(key.into(), value.into());
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.credential_type != "api_key" {
            bail!("unsupported provider credential type");
        }
        if self.key.trim().is_empty() {
            bail!("provider API key is empty");
        }
        Ok(())
    }
}

impl fmt::Debug for ProviderApiKeyCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderApiKeyCredential")
            .field("credential_type", &self.credential_type)
            .field("key", &"[REDACTED]")
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProviderStoredCredential {
    OAuth(ProviderCredential),
    ApiKey(ProviderApiKeyCredential),
}

impl ProviderStoredCredential {
    pub(crate) fn oauth(&self) -> Option<&ProviderCredential> {
        match self {
            Self::OAuth(credential) => Some(credential),
            Self::ApiKey(_) => None,
        }
    }

    pub(crate) fn metadata(&self, key: &str) -> Option<&Value> {
        match self {
            Self::OAuth(credential) => credential.metadata(key),
            Self::ApiKey(credential) => credential.metadata(key),
        }
    }

    pub(crate) fn metadata_str(&self, key: &str) -> Option<&str> {
        match self {
            Self::OAuth(credential) => credential.metadata_str(key),
            Self::ApiKey(credential) => credential.metadata_str(key),
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::OAuth(credential) => credential.validate(),
            Self::ApiKey(credential) => credential.validate(),
        }
    }

    fn to_raw(&self) -> anyhow::Result<Value> {
        match self {
            Self::OAuth(credential) => Ok(serde_json::to_value(credential)?),
            Self::ApiKey(credential) => Ok(serde_json::to_value(credential)?),
        }
    }
}

impl From<ProviderCredential> for ProviderStoredCredential {
    fn from(value: ProviderCredential) -> Self {
        Self::OAuth(value)
    }
}

impl From<ProviderApiKeyCredential> for ProviderStoredCredential {
    fn from(value: ProviderApiKeyCredential) -> Self {
        Self::ApiKey(value)
    }
}

/// State of one provider-owned credential slot. A present-but-unrecognized or
/// malformed value deliberately remains distinct from `Missing`: only the
/// latter permits provider environment fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProviderSlotState {
    Missing,
    Known(ProviderStoredCredential),
    PresentUnsupportedOrInvalid,
}

impl ProviderSlotState {
    pub(crate) fn known(&self) -> Option<&ProviderStoredCredential> {
        match self {
            Self::Known(credential) => Some(credential),
            Self::Missing | Self::PresentUnsupportedOrInvalid => None,
        }
    }

    pub(crate) fn oauth(&self) -> Option<&ProviderCredential> {
        self.known()?.oauth()
    }

    pub(crate) fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProviderStore {
    path: PathBuf,
    auth_json_path: PathBuf,
}

impl Default for ProviderStore {
    fn default() -> Self {
        Self::new(&crate::util::grok_home::grok_home())
    }
}

impl ProviderStore {
    pub(crate) fn new(grok_home: &Path) -> Self {
        let auth_json_path = std::env::var_os("GROK_AUTH_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| grok_home.join("auth.json"));
        Self {
            path: grok_home.join("providers.json"),
            auth_json_path,
        }
    }

    pub(crate) fn with_paths(path: PathBuf, auth_json_path: PathBuf) -> Self {
        Self {
            path,
            auth_json_path,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn get(&self, provider: ProviderId) -> anyhow::Result<ProviderSlotState> {
        let table = read_table(&self.path)?;
        Ok(classify_slot(provider, table.get(provider.as_str())))
    }

    pub(crate) fn list(&self) -> anyhow::Result<BTreeMap<ProviderId, ProviderSlotState>> {
        let table = read_table(&self.path)?;
        Ok(table
            .iter()
            .filter_map(|(key, value)| {
                key.parse()
                    .ok()
                    .map(|provider| (provider, classify_slot(provider, Some(value))))
            })
            .collect())
    }

    pub(crate) async fn put<C>(&self, provider: ProviderId, credential: C) -> anyhow::Result<()>
    where
        C: Into<ProviderStoredCredential>,
    {
        let credential = credential.into();
        credential.validate()?;
        let raw = credential.to_raw()?;
        let _process_guard = PROCESS_MUTATION_LOCK.lock().await;
        let lock = self
            .acquire(crate::auth::manager::AUTH_LOCK_TIMEOUT)
            .await?;
        let mut table = read_table(&self.path)?;
        table.insert(provider.as_str().to_owned(), raw);
        self.ensure_live(&lock)?;
        write_table(&self.path, &table)
    }

    pub(crate) async fn remove(&self, provider: ProviderId) -> anyhow::Result<bool> {
        let _process_guard = PROCESS_MUTATION_LOCK.lock().await;
        let lock = self
            .acquire(crate::auth::manager::AUTH_LOCK_TIMEOUT)
            .await?;
        let mut table = read_table(&self.path)?;
        let removed = table.remove(provider.as_str()).is_some();
        if removed {
            self.ensure_live(&lock)?;
            write_or_remove_table(&self.path, &table)?;
        }
        Ok(removed)
    }

    pub(crate) async fn clear(&self) -> anyhow::Result<usize> {
        let _process_guard = PROCESS_MUTATION_LOCK.lock().await;
        let lock = self
            .acquire(crate::auth::manager::AUTH_LOCK_TIMEOUT)
            .await?;
        let count = read_table(&self.path)?.len();
        if count > 0 {
            self.ensure_live(&lock)?;
            remove_file_if_present(&self.path)?;
        }
        Ok(count)
    }

    /// Refresh under the shared `auth.json.lock`. The table is re-read after
    /// locking and checked again, so a sibling process that already rotated a
    /// refresh token wins and the stale token is never spent twice.
    pub(crate) async fn refresh<F, Fut>(
        &self,
        provider: ProviderId,
        reason: RefreshReason,
        refresher: F,
    ) -> anyhow::Result<ProviderCredential>
    where
        F: FnOnce(ProviderCredential) -> Fut,
        Fut: Future<Output = anyhow::Result<ProviderRefreshOutcome>>,
    {
        let _process_guard = PROCESS_MUTATION_LOCK.lock().await;
        let lock = self
            .acquire(crate::auth::manager::REFRESH_LOCK_TIMEOUT)
            .await?;
        let mut table = read_table(&self.path)?;
        let current = match classify_slot(provider, table.get(provider.as_str())) {
            ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => credential,
            ProviderSlotState::Known(ProviderStoredCredential::ApiKey(_)) => {
                bail!("{provider} stores an API key, which cannot be refreshed")
            }
            ProviderSlotState::Missing => bail!("{provider} is not logged in"),
            ProviderSlotState::PresentUnsupportedOrInvalid => {
                bail!("{provider} credential slot is unsupported or invalid")
            }
        };
        current.validate()?;

        match &reason {
            RefreshReason::Expiring if !current.expires_within(REFRESH_WINDOW) => {
                return Ok(current);
            }
            RefreshReason::Rejected { access } if current.access != *access => {
                return Ok(current);
            }
            RefreshReason::Expiring | RefreshReason::Rejected { .. } => {}
        }

        if current.refresh.is_empty() {
            bail!(
                "{provider} credential cannot be refreshed; run `grok login --provider {provider}`"
            );
        }

        // Never send a rotating refresh token after stale-lock recovery has
        // moved the live lock to a new inode.
        self.ensure_live(&lock)?;
        let outcome = refresher(current).await?;
        self.ensure_live(&lock)?;

        match outcome {
            ProviderRefreshOutcome::Save(credential) => {
                credential.validate()?;
                table.insert(
                    provider.as_str().to_owned(),
                    ProviderStoredCredential::OAuth(credential.clone()).to_raw()?,
                );
                write_table(&self.path, &table)?;
                Ok(credential)
            }
            ProviderRefreshOutcome::Remove { message } => {
                table.remove(provider.as_str());
                write_or_remove_table(&self.path, &table)?;
                bail!("{message}")
            }
        }
    }

    async fn acquire(
        &self,
        timeout: Duration,
    ) -> anyhow::Result<crate::auth::storage::AuthFileLock> {
        crate::auth::manager::lock::try_lock_auth_file_async(&self.auth_json_path, timeout)
            .await
            .with_context(|| {
                format!(
                    "timed out acquiring credential lock {}",
                    self.auth_json_path
                        .with_file_name("auth.json.lock")
                        .display()
                )
            })
    }

    fn ensure_live(&self, lock: &crate::auth::storage::AuthFileLock) -> anyhow::Result<()> {
        if lock.still_live(&self.auth_json_path) {
            Ok(())
        } else {
            bail!("credential lock was replaced while held; retrying avoids token corruption")
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RefreshReason {
    Expiring,
    Rejected { access: String },
}

pub(crate) fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn classify_slot(provider: ProviderId, raw: Option<&Value>) -> ProviderSlotState {
    let Some(raw) = raw else {
        return ProviderSlotState::Missing;
    };
    let credential_type = raw
        .as_object()
        .and_then(|object| object.get("type"))
        .and_then(Value::as_str);
    let parsed = match credential_type {
        Some("oauth") => serde_json::from_value::<ProviderCredential>(raw.clone())
            .ok()
            .and_then(|credential| credential.validate().ok().map(|()| credential))
            .map(ProviderStoredCredential::OAuth),
        Some("api_key") => serde_json::from_value::<ProviderApiKeyCredential>(raw.clone())
            .ok()
            .and_then(|credential| credential.validate().ok().map(|()| credential))
            .map(ProviderStoredCredential::ApiKey),
        Some(_) | None => None,
    };
    match parsed {
        Some(credential) => ProviderSlotState::Known(credential),
        None => {
            let type_class = match credential_type {
                Some("oauth") => "oauth",
                Some("api_key") => "api_key",
                Some(_) => "unsupported",
                None => "missing_or_non_string",
            };
            tracing::warn!(
                %provider,
                credential_type = type_class,
                classification = "present_unsupported_or_invalid",
                "provider credential slot is not usable; environment fallback is blocked"
            );
            ProviderSlotState::PresentUnsupportedOrInvalid
        }
    }
}

fn read_table(path: &Path) -> anyhow::Result<RawCredentialTable> {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RawCredentialTable::new());
        }
        Err(error) => return Err(error.into()),
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    if let Err(error) = crate::util::secure_file::ensure_owner_only_permissions(path) {
        tracing::warn!(path = %path.display(), %error, "provider auth: failed to tighten providers.json permissions");
    }
    if contents.trim().is_empty() {
        return Ok(RawCredentialTable::new());
    }
    let value: Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    match value {
        Value::Object(table) => Ok(table),
        _ => bail!("failed to parse {}: root must be a JSON object", path.display()),
    }
}

fn write_or_remove_table(path: &Path, table: &RawCredentialTable) -> anyhow::Result<()> {
    if table.is_empty() {
        remove_file_if_present(path)
    } else {
        write_table(path, table)
    }
}

fn remove_file_if_present(path: &Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => sync_parent(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn write_table(path: &Path, table: &RawCredentialTable) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!(
        "json.{}.{}.tmp",
        std::process::id(),
        TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    struct Reclaim(Option<PathBuf>);
    impl Drop for Reclaim {
        fn drop(&mut self) {
            if let Some(path) = self.0.as_deref() {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    let mut reclaim = Reclaim(Some(tmp.clone()));

    let file = crate::util::secure_file::open_secure_file(&tmp)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, table)?;
    writer.flush()?;
    writer
        .into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;

    #[cfg(windows)]
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    std::fs::rename(&tmp, path)?;
    reclaim.0 = None;
    crate::util::secure_file::ensure_owner_only_permissions(path)?;
    sync_parent(path)
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn store(dir: &tempfile::TempDir) -> ProviderStore {
        ProviderStore::with_paths(
            dir.path().join("providers.json"),
            dir.path().join("auth.json"),
        )
    }

    fn oauth(slot: ProviderSlotState) -> ProviderCredential {
        match slot {
            ProviderSlotState::Known(ProviderStoredCredential::OAuth(credential)) => credential,
            other => panic!("expected OAuth slot, got {other:?}"),
        }
    }

    fn api_key(slot: ProviderSlotState) -> ProviderApiKeyCredential {
        match slot {
            ProviderSlotState::Known(ProviderStoredCredential::ApiKey(credential)) => credential,
            other => panic!("expected API-key slot, got {other:?}"),
        }
    }

    #[test]
    fn credentials_round_trip_metadata_and_redact_debug() {
        let mut oauth_credential = ProviderCredential::oauth("access-secret", "refresh-secret", 123);
        oauth_credential.set_metadata("accountId", "acct-1");
        let oauth_json = serde_json::to_string(&oauth_credential).unwrap();
        let oauth_decoded: ProviderCredential = serde_json::from_str(&oauth_json).unwrap();
        assert_eq!(oauth_decoded.metadata_str("accountId"), Some("acct-1"));
        let oauth_debug = format!("{oauth_decoded:?}");
        assert!(!oauth_debug.contains("access-secret"));
        assert!(!oauth_debug.contains("refresh-secret"));

        let mut api_key_credential = ProviderApiKeyCredential::new("api-key-secret");
        api_key_credential.set_metadata("future", serde_json::json!({"enabled": true}));
        let api_key_json = serde_json::to_string(&api_key_credential).unwrap();
        let api_key_decoded: ProviderApiKeyCredential =
            serde_json::from_str(&api_key_json).unwrap();
        assert_eq!(
            api_key_decoded.metadata("future"),
            Some(&serde_json::json!({"enabled": true}))
        );
        assert!(!format!("{api_key_decoded:?}").contains("api-key-secret"));
        assert!(!format!("{:?}", ProviderStoredCredential::ApiKey(api_key_decoded))
            .contains("api-key-secret"));
    }

    #[tokio::test]
    async fn api_key_round_trip_overwrite_remove_and_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store
            .put(
                ProviderId::Anthropic,
                ProviderCredential::oauth("oauth-access", "oauth-refresh", u64::MAX),
            )
            .await
            .unwrap();
        store
            .put(
                ProviderId::Anthropic,
                ProviderApiKeyCredential::new("stored-api-key"),
            )
            .await
            .unwrap();
        assert_eq!(api_key(store.get(ProviderId::Anthropic).unwrap()).key, "stored-api-key");
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(raw.as_object().unwrap().len(), 1);
        assert_eq!(raw["anthropic"]["type"], "api_key");
        assert!(raw["anthropic"].get("access").is_none());

        store
            .put(
                ProviderId::Anthropic,
                ProviderCredential::oauth("last-access", "last-refresh", u64::MAX),
            )
            .await
            .unwrap();
        assert_eq!(oauth(store.get(ProviderId::Anthropic).unwrap()).access, "last-access");
        assert!(store.remove(ProviderId::Anthropic).await.unwrap());
        assert!(!store.path().exists());
        assert!(!dir.path().join("auth.json").exists());
    }

    #[tokio::test]
    async fn raw_rmw_preserves_unknown_entries_and_future_fields_semantically() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let original = serde_json::json!({
            "future-provider": {
                "type": "future_credential",
                "nested": {"array": [1, true, {"secret": "opaque"}]}
            },
            "anthropic": {
                "type": "oauth-v2",
                "access": "must-not-be-treated-as-oauth",
                "refresh": "opaque",
                "expires": 42,
                "future": ["kept"]
            },
            "radius": {
                "type": "api_key",
                "key": "radius-key",
                "futureMetadata": {"region": "mars"}
            }
        });
        std::fs::write(store.path(), serde_json::to_vec_pretty(&original).unwrap()).unwrap();

        assert_eq!(
            store.get(ProviderId::Anthropic).unwrap(),
            ProviderSlotState::PresentUnsupportedOrInvalid
        );
        store
            .put(
                ProviderId::Openrouter,
                ProviderApiKeyCredential::new("openrouter-key"),
            )
            .await
            .unwrap();
        let after_put: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        for key in ["future-provider", "anthropic", "radius"] {
            assert_eq!(after_put[key], original[key]);
        }

        assert!(store.remove(ProviderId::Openrouter).await.unwrap());
        let after_remove: Value =
            serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(after_remove, original);
    }

    #[test]
    fn invalid_root_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        std::fs::write(store.path(), "[]").unwrap();
        assert!(store.get(ProviderId::Anthropic).is_err());
        std::fs::write(store.path(), "{").unwrap();
        assert!(store.get(ProviderId::Anthropic).is_err());
    }

    #[tokio::test]
    async fn api_key_refresh_never_invokes_callback() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store
            .put(
                ProviderId::Anthropic,
                ProviderApiKeyCredential::new("stored-api-key"),
            )
            .await
            .unwrap();
        let calls = AtomicUsize::new(0);
        let result = store
            .refresh(ProviderId::Anthropic, RefreshReason::Expiring, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { panic!("API-key refresh callback must not run") }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn concurrent_refresh_spends_rotating_token_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store
            .put(
                ProviderId::Anthropic,
                ProviderCredential::oauth("old-access", "old-refresh", 0),
            )
            .await
            .unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let refresh_once = |store: ProviderStore, calls: Arc<AtomicUsize>| async move {
            store
                .refresh(ProviderId::Anthropic, RefreshReason::Expiring, move |old| {
                    let calls = calls.clone();
                    async move {
                        assert_eq!(old.refresh, "old-refresh");
                        calls.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Ok(ProviderRefreshOutcome::Save(ProviderCredential::oauth(
                            "new-access",
                            "new-refresh",
                            unix_millis() + 3_600_000,
                        )))
                    }
                })
                .await
                .unwrap()
        };

        let (left, right) = tokio::join!(
            refresh_once(store.clone(), calls.clone()),
            refresh_once(store.clone(), calls.clone())
        );
        assert_eq!(left.access, "new-access");
        assert_eq!(right.access, "new-access");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejected_old_access_adopts_sibling_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store
            .put(
                ProviderId::OpenaiCodex,
                ProviderCredential::oauth("new-access", "new-refresh", u64::MAX),
            )
            .await
            .unwrap();
        let credential = store
            .refresh(
                ProviderId::OpenaiCodex,
                RefreshReason::Rejected {
                    access: "old-access".to_owned(),
                },
                |_| async { panic!("already-rotated token must not refresh again") },
            )
            .await
            .unwrap();
        assert_eq!(credential.access, "new-access");
    }

    #[test]
    #[ignore = "subprocess entry point for concurrent refresh test"]
    fn subprocess_provider_refresh() {
        let Some(root) = std::env::var_os("GROK_TEST_PROVIDER_REFRESH_DIR") else {
            return;
        };
        let root = PathBuf::from(root);
        let store = ProviderStore::with_paths(root.join("providers.json"), root.join("auth.json"));
        let counter = root.join("refresh-spends.txt");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                store
                    .refresh(
                        ProviderId::Anthropic,
                        RefreshReason::Expiring,
                        |old| async move {
                            assert_eq!(old.refresh, "rotating-refresh");
                            let mut file = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(counter)
                                .unwrap();
                            writeln!(file, "spent").unwrap();
                            file.sync_all().unwrap();
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            Ok(ProviderRefreshOutcome::Save(ProviderCredential::oauth(
                                "rotated-access",
                                "rotated-refresh",
                                unix_millis() + 3_600_000,
                            )))
                        },
                    )
                    .await
                    .unwrap();
            });
    }

    #[tokio::test]
    async fn two_processes_spend_rotating_refresh_token_once() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store
            .put(
                ProviderId::Anthropic,
                ProviderCredential::oauth("old-access", "rotating-refresh", 0),
            )
            .await
            .unwrap();

        let spawn = || {
            #[allow(clippy::disallowed_methods)] // isolated test subprocess
            std::process::Command::new(std::env::current_exe().unwrap())
                .env("GROK_TEST_PROVIDER_REFRESH_DIR", dir.path())
                .args([
                    "--ignored",
                    "--exact",
                    "--nocapture",
                    "auth::providers::store::tests::subprocess_provider_refresh",
                ])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        };
        let first = spawn();
        let second = spawn();
        let first = first.wait_with_output().unwrap();
        let second = second.wait_with_output().unwrap();
        assert!(
            first.status.success(),
            "first refresh child failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&first.stdout),
            String::from_utf8_lossy(&first.stderr)
        );
        assert!(
            second.status.success(),
            "second refresh child failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&second.stdout),
            String::from_utf8_lossy(&second.stderr)
        );

        let spends = std::fs::read_to_string(dir.path().join("refresh-spends.txt")).unwrap();
        assert_eq!(spends.lines().count(), 1);
        let stored = oauth(store.get(ProviderId::Anthropic).unwrap());
        assert_eq!(stored.access, "rotated-access");
        assert_eq!(stored.refresh, "rotated-refresh");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn providers_file_is_owner_only_for_api_keys() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(
            store.path(),
            r#"{"radius":{"type":"api_key","key":"sentinel"}}"#,
        )
        .unwrap();
        let _ = store.get(ProviderId::Radius).unwrap();
        let tightened_mode = std::fs::metadata(store.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(tightened_mode & 0o777, 0o600);

        store
            .put(
                ProviderId::Radius,
                ProviderApiKeyCredential::new("replacement"),
            )
            .await
            .unwrap();
        let written_mode = std::fs::metadata(store.path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(written_mode & 0o777, 0o600);
    }
}
