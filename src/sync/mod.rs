use std::future::Future;

use crate::chunk::now_epoch_secs;

/// A signed record mapping a name to a content hash.
/// Names follow the pattern `did:key:z6Mk.../perspectives/decisions` (a DID + path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameRecord {
    /// The stable name (e.g. "did:key:z6Mk.../perspectives/decisions")
    pub name: String,
    /// The current content hash this name points to
    pub hash: String,
    /// Unix timestamp when this record was published
    pub published_at: i64,
    /// Unix timestamp when this record expires (0 = never)
    pub expires_at: i64,
}

/// A write lease granting exclusive write access to a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    /// The name this lease is for
    pub name: String,
    /// Unique lease identifier
    pub lease_id: String,
    /// Unix timestamp when the lease expires
    pub expires_at: i64,
}

impl Lease {
    /// Check if this lease has expired.
    pub fn is_expired(&self) -> bool {
        now_epoch_secs() >= self.expires_at
    }
}

/// Remote blob storage for sync operations.
///
/// Implementations handle the transport (iroh, S3, HTTP, etc.).
/// All data is opaque bytes — encryption is the caller's responsibility.
///
/// The `hash` parameter is a blake3 hash used as the content address.
/// In the distributed case, this will be the hash of the ciphertext (not plaintext).
pub trait SyncBackend: Send + Sync {
    /// Push bytes to remote storage at the given hash.
    /// Idempotent: pushing the same hash twice is a no-op.
    fn push(
        &self,
        hash: blake3::Hash,
        data: Vec<u8>,
    ) -> impl Future<Output = crate::Result<()>> + Send;

    /// Pull bytes from remote storage by hash.
    /// Returns None if the hash doesn't exist remotely.
    fn pull(
        &self,
        hash: blake3::Hash,
    ) -> impl Future<Output = crate::Result<Option<Vec<u8>>>> + Send;

    /// Check if a hash exists in remote storage.
    fn has(&self, hash: blake3::Hash) -> impl Future<Output = crate::Result<bool>> + Send;

    /// List all hashes available in remote storage.
    fn list_hashes(&self) -> impl Future<Output = crate::Result<Vec<blake3::Hash>>> + Send;

    /// Delete a hash from remote storage.
    fn delete(&self, hash: blake3::Hash) -> impl Future<Output = crate::Result<()>> + Send;

    /// Human-readable name for this backend (e.g. "iroh", "s3", "http").
    fn name(&self) -> &str;
}

/// Name resolution for distributed VecLayer stores.
///
/// Maps stable names (DID + path) to current content hashes.
/// Implementations handle the resolution backend (Postgres, edge, P2P DHT).
pub trait NameResolver: Send + Sync {
    /// Resolve a name to its current record.
    /// Returns None if the name has never been published.
    fn resolve(&self, name: &str)
        -> impl Future<Output = crate::Result<Option<NameRecord>>> + Send;

    /// Acquire an exclusive write lease for a name.
    /// Returns an error if the name is already leased by another writer.
    fn acquire_lease(&self, name: &str) -> impl Future<Output = crate::Result<Lease>> + Send;

    /// Release a previously acquired lease.
    fn release_lease(&self, lease: &Lease) -> impl Future<Output = crate::Result<()>> + Send;

    /// Publish a new record for a name. Requires a valid lease.
    fn publish(
        &self,
        lease: &Lease,
        record: NameRecord,
    ) -> impl Future<Output = crate::Result<()>> + Send;

    /// Human-readable name for this resolver (e.g. "postgres", "cloudflare", "dht").
    fn name(&self) -> &str;
}

crate::arc_impl!(SyncBackend {
    fn push(&self, hash: blake3::Hash, data: Vec<u8>) -> impl Future<Output = crate::Result<()>> + Send;
    fn pull(&self, hash: blake3::Hash) -> impl Future<Output = crate::Result<Option<Vec<u8>>>> + Send;
    fn has(&self, hash: blake3::Hash) -> impl Future<Output = crate::Result<bool>> + Send;
    fn list_hashes(&self) -> impl Future<Output = crate::Result<Vec<blake3::Hash>>> + Send;
    fn delete(&self, hash: blake3::Hash) -> impl Future<Output = crate::Result<()>> + Send;
    fn name(&self) -> &str;
});

crate::arc_impl!(NameResolver {
    fn resolve(&self, name: &str) -> impl Future<Output = crate::Result<Option<NameRecord>>> + Send;
    fn acquire_lease(&self, name: &str) -> impl Future<Output = crate::Result<Lease>> + Send;
    fn release_lease(&self, lease: &Lease) -> impl Future<Output = crate::Result<()>> + Send;
    fn publish(&self, lease: &Lease, record: NameRecord) -> impl Future<Output = crate::Result<()>> + Send;
    fn name(&self) -> &str;
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_record_equality() {
        let a = NameRecord {
            name: "did:key:z6Mk.../decisions".to_string(),
            hash: "abc123".to_string(),
            published_at: 1000,
            expires_at: 0,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn lease_expiry() {
        let expired = Lease {
            name: "test".to_string(),
            lease_id: "lease-1".to_string(),
            expires_at: 0,
        };
        assert!(expired.is_expired());

        let valid = Lease {
            name: "test".to_string(),
            lease_id: "lease-2".to_string(),
            expires_at: i64::MAX,
        };
        assert!(!valid.is_expired());
    }

    struct MockSync;

    impl SyncBackend for MockSync {
        fn push(
            &self,
            _hash: blake3::Hash,
            _data: Vec<u8>,
        ) -> impl Future<Output = crate::Result<()>> + Send {
            async { Ok(()) }
        }

        fn pull(
            &self,
            _hash: blake3::Hash,
        ) -> impl Future<Output = crate::Result<Option<Vec<u8>>>> + Send {
            async { Ok(None) }
        }

        fn has(&self, _hash: blake3::Hash) -> impl Future<Output = crate::Result<bool>> + Send {
            async { Ok(false) }
        }

        fn list_hashes(&self) -> impl Future<Output = crate::Result<Vec<blake3::Hash>>> + Send {
            async { Ok(vec![]) }
        }

        fn delete(&self, _hash: blake3::Hash) -> impl Future<Output = crate::Result<()>> + Send {
            async { Ok(()) }
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn mock_sync_compiles_and_runs() {
        let sync = MockSync;
        let hash = blake3::hash(b"test");
        assert!(!sync.has(hash).await.unwrap());
        assert_eq!(sync.name(), "mock");
    }

    struct MockResolver;

    impl NameResolver for MockResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> impl Future<Output = crate::Result<Option<NameRecord>>> + Send {
            async { Ok(None) }
        }

        fn acquire_lease(&self, name: &str) -> impl Future<Output = crate::Result<Lease>> + Send {
            let name = name.to_string();
            async move {
                Ok(Lease {
                    name,
                    lease_id: "test-lease".to_string(),
                    expires_at: i64::MAX,
                })
            }
        }

        fn release_lease(&self, _lease: &Lease) -> impl Future<Output = crate::Result<()>> + Send {
            async { Ok(()) }
        }

        fn publish(
            &self,
            _lease: &Lease,
            _record: NameRecord,
        ) -> impl Future<Output = crate::Result<()>> + Send {
            async { Ok(()) }
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn mock_resolver_compiles_and_runs() {
        let resolver = MockResolver;
        let result = resolver.resolve("did:key:test/decisions").await.unwrap();
        assert!(result.is_none());

        let lease = resolver.acquire_lease("test").await.unwrap();
        assert!(!lease.is_expired());
        assert_eq!(resolver.name(), "mock");
    }

    #[tokio::test]
    async fn arc_sync_backend_works() {
        let sync: std::sync::Arc<MockSync> = std::sync::Arc::new(MockSync);
        let hash = blake3::hash(b"test");
        assert!(!sync.has(hash).await.unwrap());
    }

    #[tokio::test]
    async fn arc_name_resolver_works() {
        let resolver: std::sync::Arc<MockResolver> = std::sync::Arc::new(MockResolver);
        let result = resolver.resolve("test").await.unwrap();
        assert!(result.is_none());
    }
}
