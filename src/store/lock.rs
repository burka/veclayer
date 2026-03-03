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
    #[cfg(test)]
    pub fn acquire(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        let lock_path = data_dir.join(LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        file.try_lock_exclusive().map_err(|e| {
            if e.kind() == std::io::ErrorKind::WouldBlock {
                Error::store(
                    "Another VecLayer process is writing to this store. \
                     Use --read-only for concurrent read access.",
                )
            } else {
                Error::store(format!("Failed to acquire store lock: {}", e))
            }
        })?;

        Ok(Self { _file: file })
    }

    /// Acquire an exclusive lock on `data_dir`, blocking until available or timeout.
    ///
    /// Retries with exponential backoff for up to 2 seconds. Returns a clear
    /// error identifying the lock file if another process holds it too long.
    pub fn acquire_blocking(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        let lock_path = data_dir.join(LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;

        // Retry with backoff: 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1280ms ≈ 2.5s total
        let mut wait = std::time::Duration::from_millis(10);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);

        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() + wait > deadline {
                        return Err(Error::store(format!(
                            "Timed out waiting for write lock on {} — \
                             another veclayer process may be holding it. \
                             Check for stale processes: lsof {}",
                            lock_path.display(),
                            lock_path.display(),
                        )));
                    }
                    std::thread::sleep(wait);
                    wait = std::time::Duration::from_millis((wait.as_millis() as u64 * 2).min(320));
                }
                Err(e) => {
                    return Err(Error::store(format!(
                        "Failed to acquire store lock {}: {}",
                        lock_path.display(),
                        e
                    )));
                }
            }
        }
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

    #[test]
    fn test_lock_timeout() {
        let dir = TempDir::new().unwrap();
        let _lock = FileLock::acquire(dir.path()).unwrap();

        // Second acquisition should timeout (not hang forever)
        let start = std::time::Instant::now();
        let result = FileLock::acquire_blocking(dir.path());
        let elapsed = start.elapsed();

        assert!(result.is_err());
        assert!(elapsed.as_secs() >= 1, "should have retried for ~2s");
        assert!(elapsed.as_secs() <= 5, "should not hang forever");

        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Timed out"), "unexpected: {msg}");
    }

    /// Verify that a read-only directory (mode 0o444) causes acquire_blocking to
    /// return a clear error rather than panicking. The lock file cannot be created
    /// because the directory is not writable.
    #[cfg(unix)]
    #[test]
    fn test_lock_acquire_permission_denied() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();

        // Make the directory read-only so no file can be created inside it.
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o444)).unwrap();

        let result = FileLock::acquire_blocking(dir.path());

        // Restore permissions so TempDir cleanup succeeds.
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            result.is_err(),
            "acquire_blocking on a read-only directory must return Err, not panic"
        );
        // The error may come from opening the lock file (OS-level "permission denied")
        // or from locking; either way it must be an Err — just verify it is not Ok.
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("permission denied")
                || msg.contains("Failed to acquire store lock"),
            "unexpected error message: {msg}"
        );
    }

    /// Verify the POSIX flock guarantee: the lock is released when the file
    /// descriptor is closed (i.e. when the holding thread exits and the
    /// FileLock value is dropped). After joining the thread, a new acquisition
    /// on the same directory must succeed immediately.
    #[test]
    fn test_lock_released_on_thread_exit() {
        let dir = TempDir::new().unwrap();
        let dir_path = dir.path().to_path_buf();

        let handle = std::thread::spawn(move || {
            // Acquire the lock inside the thread, then let it drop when the
            // thread function returns.
            let _lock = FileLock::acquire_blocking(&dir_path).unwrap();
            // Lock is held here and released on function exit.
        });

        handle.join().expect("thread must not panic");

        // The thread has exited and dropped FileLock; re-acquiring must succeed.
        let result = FileLock::acquire_blocking(dir.path());
        assert!(
            result.is_ok(),
            "lock must be released after the holding thread exits: {:?}",
            result.err()
        );
    }
}
