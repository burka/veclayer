//! Binary embedding cache I/O for the git memory branch.
//!
//! Embeddings are stored as raw little-endian `f32` blobs under `.embeddings/`
//! on the memory branch:
//!
//! ```text
//! .embeddings/<short-id>.<model-name>.bin
//! ```
//!
//! A 384-dimension embedding occupies exactly 1,536 bytes (384 × 4).
//! Reads use `git show` via plumbing; writes go through the worktree.

use super::{is_file_not_found, GitError, GitMemoryBranch};

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

impl GitMemoryBranch {
    /// Return the relative path for an embedding blob on the memory branch.
    ///
    /// Format: `.embeddings/<id>.<model>.bin`
    pub fn embedding_path(id: &str, model: &str) -> String {
        format!(".embeddings/{id}.{model}.bin")
    }
}

// ---------------------------------------------------------------------------
// Read / write
// ---------------------------------------------------------------------------

impl GitMemoryBranch {
    /// Read a cached embedding for `id` / `model` from the memory branch.
    ///
    /// Returns `Ok(None)` when no `.bin` file exists for the given combination
    /// (not an error — the caller should compute the embedding instead).
    /// Returns `Ok(Some(vec))` with the decoded `f32` values on success.
    ///
    /// The blob is decoded as little-endian `f32` values.
    pub fn read_embedding(&self, id: &str, model: &str) -> Result<Option<Vec<f32>>, GitError> {
        let path = Self::embedding_path(id, model);

        match self.read_file(&path) {
            Ok(bytes) => Ok(Some(bytes_to_floats(&bytes))),
            Err(GitError::CommandFailed { ref stderr, .. }) if is_file_not_found(stderr) => {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Write `vector` as a raw little-endian `f32` blob to the memory branch worktree.
    ///
    /// The file is placed at `.embeddings/<id>.<model>.bin`. Parent directories
    /// are created automatically by `write_file`. The caller is responsible for
    /// committing and pushing afterwards.
    pub fn write_embedding(&self, id: &str, model: &str, vector: &[f32]) -> Result<(), GitError> {
        let path = Self::embedding_path(id, model);
        let bytes = floats_to_bytes(vector);
        self.write_file(&path, &bytes)
    }
}

// ---------------------------------------------------------------------------
// List / GC
// ---------------------------------------------------------------------------

impl GitMemoryBranch {
    /// List all entry IDs that have a cached embedding for `model`.
    ///
    /// Scans the memory branch via `list_files()` and extracts the ID portion
    /// from each matching `.embeddings/*.<model>.bin` filename.
    pub fn list_cached_embeddings(&self, model: &str) -> Result<Vec<String>, GitError> {
        let suffix = format!(".{model}.bin");
        let prefix = ".embeddings/";

        self.list_files().map(|files| {
            files
                .into_iter()
                .filter_map(|path| extract_embedding_id(&path, prefix, &suffix))
                .collect()
        })
    }

    /// Delete `.bin` files whose ID is not in `live_ids` for the given `model`.
    ///
    /// Returns the count of deleted files. Does not commit — the caller should
    /// commit and push (with `--force-with-lease` if desired) afterwards.
    pub fn gc_embeddings(&self, live_ids: &[&str], model: &str) -> Result<usize, GitError> {
        use std::collections::HashSet;

        let cached = self.list_cached_embeddings(model)?;
        let live: HashSet<&str> = live_ids.iter().copied().collect();
        let mut deleted = 0;

        for id in cached {
            if !live.contains(id.as_str()) {
                let path = Self::embedding_path(&id, model);
                self.delete_file(&path)?;
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Encode a slice of `f32` values as a little-endian byte vector.
fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Decode a byte slice into a `Vec<f32>` using little-endian byte order.
///
/// Logs a warning if the byte slice length is not a multiple of 4 (trailing
/// bytes are discarded).
fn bytes_to_floats(bytes: &[u8]) -> Vec<f32> {
    if !bytes.len().is_multiple_of(4) {
        tracing::warn!(
            "embedding blob has {} trailing bytes (total {} bytes, expected multiple of 4)",
            bytes.len() % 4,
            bytes.len(),
        );
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Extract the entry ID from an embedding file path.
///
/// Expects the form `<prefix><id><suffix>`, e.g.
/// `.embeddings/bf81639.bge-small-en-v1.5.bin`.
fn extract_embedding_id(path: &str, prefix: &str, suffix: &str) -> Option<String> {
    let after_prefix = path.strip_prefix(prefix)?;
    let id = after_prefix.strip_suffix(suffix)?;
    if id.is_empty() {
        return None;
    }
    Some(id.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn setup_test_repo_with_branch() -> (tempfile::TempDir, GitMemoryBranch) {
        let dir = tempfile::tempdir().unwrap();
        let git_dir = dir.path().join(".git");

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        std::process::Command::new("git")
            .args(["-c", "user.email=test@test.com", "-c", "user.name=Test"])
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let branch = GitMemoryBranch::open(&git_dir, Some("test-memory")).unwrap();
        branch.create_orphan_branch().unwrap();

        (dir, branch)
    }

    // -----------------------------------------------------------------------
    // embedding_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_path() {
        let path = GitMemoryBranch::embedding_path("bf81639", "bge-small-en-v1.5");
        assert_eq!(path, ".embeddings/bf81639.bge-small-en-v1.5.bin");
    }

    // -----------------------------------------------------------------------
    // write_embedding / read_embedding
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_and_read_embedding() {
        let (_dir, branch) = setup_test_repo_with_branch();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        branch
            .write_embedding("abc1234", "test-model", &vector)
            .unwrap();
        branch.commit("add embedding").unwrap();

        let read_back = branch
            .read_embedding("abc1234", "test-model")
            .unwrap()
            .expect("embedding should be present");

        assert_eq!(read_back.len(), vector.len());
        for (got, expected) in read_back.iter().zip(vector.iter()) {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "value mismatch: got {got}, expected {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // read_embedding — missing file
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_missing_embedding() {
        let (_dir, branch) = setup_test_repo_with_branch();

        let result = branch.read_embedding("nosuchid", "test-model").unwrap();

        assert!(result.is_none(), "expected None for missing embedding");
    }

    // -----------------------------------------------------------------------
    // list_cached_embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_cached_embeddings() {
        let (_dir, branch) = setup_test_repo_with_branch();

        let vector: Vec<f32> = vec![1.0_f32; 8];

        branch
            .write_embedding("aaa1111", "mymodel", &vector)
            .unwrap();
        branch
            .write_embedding("bbb2222", "mymodel", &vector)
            .unwrap();
        branch
            .write_embedding("ccc3333", "othermodel", &vector)
            .unwrap();
        branch.commit("add embeddings").unwrap();

        let mut ids = branch.list_cached_embeddings("mymodel").unwrap();
        ids.sort();

        assert_eq!(ids, vec!["aaa1111", "bbb2222"]);
    }

    // -----------------------------------------------------------------------
    // gc_embeddings
    // -----------------------------------------------------------------------

    #[test]
    fn test_gc_removes_orphans() {
        let (_dir, branch) = setup_test_repo_with_branch();

        let vector: Vec<f32> = vec![0.5_f32; 4];

        branch
            .write_embedding("live001", "gcmodel", &vector)
            .unwrap();
        branch
            .write_embedding("live002", "gcmodel", &vector)
            .unwrap();
        branch
            .write_embedding("orphan1", "gcmodel", &vector)
            .unwrap();
        branch.commit("add three embeddings").unwrap();

        let deleted = branch
            .gc_embeddings(&["live001", "live002"], "gcmodel")
            .unwrap();

        assert_eq!(deleted, 1, "exactly one orphan should have been deleted");

        branch.commit("gc: remove orphaned embeddings").unwrap();

        let remaining = branch.list_cached_embeddings("gcmodel").unwrap();
        assert!(!remaining.contains(&"orphan1".to_string()));
        assert!(remaining.contains(&"live001".to_string()));
        assert!(remaining.contains(&"live002".to_string()));
    }
}
