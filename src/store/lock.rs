//! Advisory file locking for single-writer safety.
//!
//! Uses POSIX flock (via `fs2`) so the lock is released automatically
//! when the process exits, even on a crash.

use std::fs::{File, OpenOptions};
use std::path::Path;

use fs2::FileExt;

use crate::{Error, Result};

const LOCK_FILE_NAME: &str = ".lock";

/// Exclusive advisory lock on a data directory.
///
/// The lock is held for the lifetime of this value and released on `Drop`.
#[derive(Debug)]
pub struct FileLock {
    _file: File,
}

impl FileLock {
    /// Acquire an exclusive lock on `data_dir`.
    ///
    /// Returns an error immediately if another process already holds the lock.
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        let lock_path = data_dir.join(LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        file.try_lock_exclusive().map_err(|_| {
            Error::store(
                "Another VecLayer process is writing to this store. \
                 Use --read-only for concurrent read access.",
            )
        })?;

        Ok(Self { _file: file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // fs2 releases the lock when the File is dropped; nothing extra needed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lock_file_created() {
        let dir = TempDir::new().unwrap();
        let _lock = FileLock::acquire(dir.path()).unwrap();
        assert!(dir.path().join(".lock").exists());
    }

    #[test]
    fn test_lock_acquire_release() {
        let dir = TempDir::new().unwrap();
        {
            let _lock = FileLock::acquire(dir.path()).unwrap();
        }
        // After drop, a new acquisition must succeed.
        FileLock::acquire(dir.path()).unwrap();
    }

    #[test]
    fn test_lock_exclusive() {
        let dir = TempDir::new().unwrap();
        let _lock = FileLock::acquire(dir.path()).unwrap();

        let result = FileLock::acquire(dir.path());
        assert!(result.is_err());

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Another VecLayer process"),
            "unexpected message: {msg}"
        );
    }
}
