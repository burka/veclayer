//! Git-style content-addressed blob store.
//!
//! Blobs are stored as files named by their BLAKE3 hash, sharded into
//! two-character prefix directories (like git's object store). Writes
//! are atomic via temp-file-then-rename.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entry::StoredBlob;

/// Monotonic counter for unique temp file names within a process.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII guard that removes a temp file on drop unless [`disarm`](TmpFileGuard::disarm) is called.
///
/// Used by [`BlobStore::put`] to guarantee temp-file cleanup when the
/// rename (or any subsequent step) fails.
struct TmpFileGuard {
    path: PathBuf,
    disarmed: bool,
}

impl TmpFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            disarmed: false,
        }
    }

    /// Cancel cleanup: the temp file was successfully promoted, so leave it in place.
    fn disarm(mut self) {
        self.disarmed = true;
    }
}

impl Drop for TmpFileGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Git-style content-addressed object store.
///
/// Layout: `{data_dir}/objects/{hash_hex[0..2]}/{hash_hex[2..]}`
///
/// Writes are atomic (temp file + rename) and idempotent: storing the same
/// blob twice is a no-op that returns the same hash.
pub struct BlobStore {
    objects_dir: PathBuf,
}

impl BlobStore {
    /// Open (or create) a BlobStore rooted at `data_dir/objects`.
    pub fn open(data_dir: &Path) -> crate::Result<Self> {
        let objects_dir = data_dir.join("objects");
        fs::create_dir_all(&objects_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&objects_dir, std::fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { objects_dir })
    }

    fn path_for(&self, hash: &blake3::Hash) -> PathBuf {
        let hex = hash.to_hex();
        self.objects_dir.join(&hex[..2]).join(&hex[2..])
    }

    /// Store a blob, returning its content hash.
    ///
    /// Idempotent: if the object already exists the hash is returned without
    /// re-writing the file.
    pub fn put(&self, blob: &StoredBlob) -> crate::Result<blake3::Hash> {
        let hash = blob.blob_hash();
        let final_path = self.path_for(&hash);

        if final_path.is_file() {
            return Ok(hash);
        }

        let shard_dir = final_path.parent().expect("path always has a parent");
        fs::create_dir_all(shard_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(shard_dir, std::fs::Permissions::from_mode(0o700))?;
        }

        let bytes = blob
            .to_bytes()
            .map_err(|e| crate::Error::store(format!("blob: serialize failed: {e}")))?;

        let pid = std::process::id();
        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut tmp_name = final_path.as_os_str().to_os_string();
        tmp_name.push(format!(".tmp.{pid}.{seq}"));
        let tmp_path = PathBuf::from(tmp_name);

        fs::write(&tmp_path, &bytes)?;
        // Guard: remove the temp file if anything after this point fails.
        let guard = TmpFileGuard::new(tmp_path.clone());
        fs::rename(&tmp_path, &final_path)?;
        guard.disarm();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&final_path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(hash)
    }

    /// Retrieve a blob by its content hash.
    ///
    /// Returns `Ok(None)` when the object is not present in the store.
    pub fn get(&self, hash: &blake3::Hash) -> crate::Result<Option<StoredBlob>> {
        let path = self.path_for(hash);

        if !path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&path)?;
        StoredBlob::from_bytes(&bytes)
            .map(Some)
            .map_err(|e| crate::Error::store(format!("blob: deserialize failed: {e}")))
    }

    /// Return `true` if the object is present in the store.
    pub fn has(&self, hash: &blake3::Hash) -> bool {
        self.path_for(hash).exists()
    }

    /// Iterate over all hashes in the store.
    ///
    /// Walks two levels deep under `objects_dir`, reconstructing each hash
    /// from `{shard}{file_stem}`. Temp files (containing `.tmp.`) are skipped.
    pub fn iter_hashes(&self) -> impl Iterator<Item = crate::Result<blake3::Hash>> {
        collect_hashes(&self.objects_dir).into_iter()
    }

    /// Count valid objects in the store.
    pub fn count(&self) -> crate::Result<usize> {
        let mut count = 0;
        for result in self.iter_hashes() {
            result?;
            count += 1;
        }
        Ok(count)
    }
}

fn collect_hashes(objects_dir: &Path) -> Vec<crate::Result<blake3::Hash>> {
    let mut results = Vec::new();

    let shards = match fs::read_dir(objects_dir) {
        Ok(it) => it,
        Err(e) => {
            results.push(Err(e.into()));
            return results;
        }
    };

    for shard_entry in shards.flatten() {
        let shard_name = shard_entry.file_name();
        let shard_str = shard_name.to_string_lossy();

        if shard_str.len() != 2 || !shard_str.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }

        let objects = match fs::read_dir(shard_entry.path()) {
            Ok(it) => it,
            Err(e) => {
                results.push(Err(e.into()));
                continue;
            }
        };

        for obj_entry in objects.flatten() {
            let obj_name = obj_entry.file_name();
            let obj_str = obj_name.to_string_lossy();

            if obj_str.contains(".tmp.") {
                continue;
            }

            let hex = format!("{shard_str}{obj_str}");
            match blake3::Hash::from_hex(&hex) {
                Ok(hash) => results.push(Ok(hash)),
                Err(e) => results.push(Err(crate::Error::store(format!(
                    "blob: invalid hash '{hex}': {e}"
                )))),
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_blob(content: &str) -> StoredBlob {
        crate::test_helpers::test_blob(content, "default")
    }

    #[test]
    fn put_and_get_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        let blob = test_blob("hello world");
        let hash = store.put(&blob).unwrap();

        let retrieved = store.get(&hash).unwrap().expect("blob should be present");
        assert_eq!(retrieved.entry.content, "hello world");
    }

    #[test]
    fn put_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        let blob = test_blob("same content");
        let hash1 = store.put(&blob).unwrap();
        let hash2 = store.put(&blob).unwrap();

        assert_eq!(hash1, hash2);

        // Only one file should exist in the shard dir
        let hex = hash1.to_hex();
        let shard_dir = store.objects_dir.join(&hex[..2]);
        let file_count = fs::read_dir(&shard_dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .map(|e| !e.file_name().to_string_lossy().contains(".tmp."))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(file_count, 1);
    }

    #[test]
    fn has_returns_true_after_put_false_for_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        let blob = test_blob("presence check");
        let hash = store.put(&blob).unwrap();

        assert!(store.has(&hash));

        // A hash that was never stored
        let unknown = blake3::hash(b"never stored");
        assert!(!store.has(&unknown));
    }

    #[test]
    fn iter_hashes_finds_all_stored_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        let h1 = store.put(&test_blob("alpha")).unwrap();
        let h2 = store.put(&test_blob("beta")).unwrap();
        let h3 = store.put(&test_blob("gamma")).unwrap();

        let mut found: Vec<blake3::Hash> = store.iter_hashes().map(|r| r.unwrap()).collect();
        found.sort_by_key(|h| h.to_hex().to_string());

        let mut expected = vec![h1, h2, h3];
        expected.sort_by_key(|h| h.to_hex().to_string());

        assert_eq!(found, expected);
    }

    /// Force `rename` to fail by pre-creating a directory at the final path.
    /// On Linux, renaming a regular file onto an existing directory returns
    /// EISDIR (or ENOTDIR on some kernels), so `put` must return an error.
    /// The invariant under test: no `.tmp.` file must survive in the shard dir
    /// after the failure.
    #[cfg(unix)]
    #[test]
    fn temp_file_cleaned_up_on_rename_failure() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        let blob = test_blob("rename-failure-cleanup");
        let hash = blob.blob_hash();
        let final_path = store.path_for(&hash);
        let shard_dir = final_path.parent().unwrap().to_owned();

        // Ensure the shard directory exists so we can create a blocker inside it.
        fs::create_dir_all(&shard_dir).unwrap();

        // Block the rename by placing a directory where the final file would land.
        fs::create_dir_all(&final_path).unwrap();

        // put() must return an error because rename fails.
        let result = store.put(&blob);
        assert!(result.is_err(), "put should fail when rename is blocked");

        // No .tmp. file must remain — that is the bug being fixed.
        let tmp_leftovers: Vec<_> = fs::read_dir(&shard_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            tmp_leftovers.is_empty(),
            "temp file leaked after rename failure: {:?}",
            tmp_leftovers.iter().map(|e| e.path()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn count_matches_number_of_puts() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();

        assert_eq!(store.count().unwrap(), 0);

        store.put(&test_blob("one")).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        store.put(&test_blob("two")).unwrap();
        assert_eq!(store.count().unwrap(), 2);

        // Duplicate should not increase count
        store.put(&test_blob("one")).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }
}
