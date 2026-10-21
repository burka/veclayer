//! Sweep of multipart-upload staging files leaked into a Lance dataset.
//!
//! `object_store`'s `LocalFileSystem` writes every multipart upload (which is
//! how Lance writes data files) to a staging file `<dest>#<N>` and renames it
//! to `<dest>` on completion. If the writing process dies mid-upload — e.g.
//! the `observe` hook is killed by its timeout while auto-compaction rewrites
//! a large fragment — the staging file is left behind. `LocalFileSystem::list`
//! deliberately hides `#<digits>` names, so Lance's own version cleanup never
//! sees (and never deletes) them: they accumulate until the disk is full.

use crate::Result;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Staging files younger than this may belong to an upload still in flight
/// (a concurrent `observe`/`store`), so they are never touched.
const IN_FLIGHT_GRACE: Duration = Duration::from_secs(60 * 60);

/// Metadata dirs whose files reference data files by name.
const METADATA_DIRS: [&str; 2] = ["_versions", "_transactions"];

/// Outcome of [`sweep_leaked_uploads`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeakedUploadSweep {
    pub files_removed: u64,
    pub bytes_reclaimed: u64,
}

/// A staging file eligible for removal.
struct LeakedUpload {
    path: PathBuf,
    stem: String,
    size: u64,
}

/// Remove leaked `<uuid>.lance#<N>` staging files from `dataset_dir/data`.
///
/// Only files older than [`IN_FLIGHT_GRACE`] whose `<uuid>.lance` stem is not
/// mentioned by any manifest or transaction file are deleted. A missing
/// `data/` dir is a no-op. Per-file delete failures are logged and skipped so
/// one bad file never blocks reclaiming the rest.
pub(crate) fn sweep_leaked_uploads(dataset_dir: &Path) -> Result<LeakedUploadSweep> {
    let mut leaks = find_old_leaked_uploads(&dataset_dir.join("data"))?;
    if leaks.is_empty() {
        return Ok(LeakedUploadSweep::default());
    }
    retain_unreferenced(dataset_dir, &mut leaks)?;

    let mut sweep = LeakedUploadSweep::default();
    for leak in leaks {
        match fs::remove_file(&leak.path) {
            Ok(()) => {
                sweep.files_removed += 1;
                sweep.bytes_reclaimed += leak.size;
            }
            Err(e) => tracing::warn!("Could not remove leaked upload {:?}: {e}", leak.path),
        }
    }
    Ok(sweep)
}

/// List staging files in `data_dir` older than the in-flight grace period.
fn find_old_leaked_uploads(data_dir: &Path) -> Result<Vec<LeakedUpload>> {
    let entries = match fs::read_dir(data_dir) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        other => other?,
    };
    let cutoff = SystemTime::now() - IN_FLIGHT_GRACE;
    let mut leaks = Vec::new();
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(stem) = file_name.to_str().and_then(leaked_upload_stem) else {
            continue;
        };
        let metadata = entry.metadata()?;
        if metadata.is_file() && metadata.modified()? < cutoff {
            leaks.push(LeakedUpload {
                path: entry.path(),
                stem: stem.to_owned(),
                size: metadata.len(),
            });
        }
    }
    Ok(leaks)
}

/// Drop every leak whose stem appears in any manifest or transaction file.
/// Unreadable metadata is an error: safety cannot be verified, so nothing is
/// deleted.
fn retain_unreferenced(dataset_dir: &Path, leaks: &mut Vec<LeakedUpload>) -> Result<()> {
    for dir in METADATA_DIRS {
        let entries = match fs::read_dir(dataset_dir.join(dir)) {
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            other => other?,
        };
        for entry in entries {
            let path = entry?.path();
            if path.is_file() {
                let bytes = fs::read(&path)?;
                leaks.retain(|leak| !contains_bytes(&bytes, leak.stem.as_bytes()));
            }
        }
    }
    Ok(())
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Returns the `<uuid>.lance` destination stem when `file_name` is an
/// `object_store` staging file for a Lance data file, else `None`.
fn leaked_upload_stem(file_name: &str) -> Option<&str> {
    let (stem, counter) = file_name.split_once('#')?;
    let is_counter = !counter.is_empty() && counter.bytes().all(|b| b.is_ascii_digit());
    let uuid = stem.strip_suffix(".lance")?;
    (is_counter && is_hyphenated_uuid(uuid)).then_some(stem)
}

/// `8-4-4-4-12` lowercase/uppercase hex groups, as Lance names data files.
fn is_hyphenated_uuid(candidate: &str) -> bool {
    const HYPHENS: [usize; 4] = [8, 13, 18, 23];
    candidate.len() == 36
        && candidate.bytes().enumerate().all(|(i, b)| {
            if HYPHENS.contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};
    use tempfile::TempDir;

    const UUID: &str = "001f5d16-e205-46e0-9f24-7b34ece9621e";
    const OTHER_UUID: &str = "0742d1bb-ac5f-4b0a-9c4a-765450e7fdb7";

    fn plant(dir: &Path, name: &str, len: usize, age: Duration) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, vec![0u8; len]).unwrap();
        let file = fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() - age).unwrap();
        path
    }

    fn dataset_with_data_dir() -> (TempDir, PathBuf) {
        let dataset = TempDir::new().unwrap();
        let data = dataset.path().join("data");
        fs::create_dir_all(&data).unwrap();
        (dataset, data)
    }

    const OLD: Duration = Duration::from_secs(2 * 60 * 60);

    #[test]
    fn test_leaked_upload_stem_matches_staging_names() {
        let name = format!("{UUID}.lance#1");
        assert_eq!(leaked_upload_stem(&name), Some(&name[..UUID.len() + 6]));
        let name = format!("{UUID}.lance#42");
        assert_eq!(leaked_upload_stem(&name), Some(&name[..UUID.len() + 6]));
    }

    #[test]
    fn test_leaked_upload_stem_rejects_other_names() {
        for name in [
            format!("{UUID}.lance"),
            format!("{UUID}.lance#"),
            format!("{UUID}.lance#1a"),
            format!("{UUID}.manifest#1"),
            format!("{UUID}x.lance#1"),
            "not-a-uuid-at-all-but-36-chars-long!.lance#1".to_string(),
            "foo.lance#1".to_string(),
            format!("{}.lance#1", UUID.replace('-', "_")),
            String::new(),
        ] {
            assert_eq!(leaked_upload_stem(&name), None, "{name:?} must not match");
        }
    }

    #[test]
    fn test_sweep_removes_old_leaked_upload_and_keeps_the_rest() {
        let (dataset, data) = dataset_with_data_dir();
        let old_leak = plant(&data, &format!("{UUID}.lance#1"), 4096, OLD);
        let fresh_leak = plant(&data, &format!("{OTHER_UUID}.lance#1"), 100, Duration::ZERO);
        let live = plant(&data, &format!("{OTHER_UUID}.lance"), 100, OLD);
        let unrelated = plant(&data, "notes.lance#1", 100, OLD);

        let sweep = sweep_leaked_uploads(dataset.path()).unwrap();

        assert_eq!(
            sweep,
            LeakedUploadSweep {
                files_removed: 1,
                bytes_reclaimed: 4096
            }
        );
        assert!(!old_leak.exists(), "old leaked upload must be removed");
        assert!(
            fresh_leak.exists(),
            "in-flight upload within grace must survive"
        );
        assert!(live.exists(), "committed data file must survive");
        assert!(unrelated.exists(), "non-matching file must survive");
    }

    #[test]
    fn test_sweep_keeps_leak_whose_stem_is_referenced_by_a_manifest() {
        let (dataset, data) = dataset_with_data_dir();
        let versions = dataset.path().join("_versions");
        fs::create_dir_all(&versions).unwrap();
        let mut manifest = b"\x0a\x12proto-prefix data/".to_vec();
        manifest.extend_from_slice(format!("{UUID}.lance").as_bytes());
        fs::write(versions.join("7.manifest"), manifest).unwrap();
        let referenced = plant(&data, &format!("{UUID}.lance#1"), 10, OLD);

        let sweep = sweep_leaked_uploads(dataset.path()).unwrap();

        assert_eq!(sweep, LeakedUploadSweep::default());
        assert!(referenced.exists(), "manifest-referenced file must survive");
    }

    #[test]
    fn test_sweep_keeps_leak_whose_stem_is_referenced_by_a_transaction() {
        let (dataset, data) = dataset_with_data_dir();
        let transactions = dataset.path().join("_transactions");
        fs::create_dir_all(&transactions).unwrap();
        fs::write(transactions.join("3-abc.txn"), format!("..{UUID}.lance..")).unwrap();
        let referenced = plant(&data, &format!("{UUID}.lance#1"), 10, OLD);

        let sweep = sweep_leaked_uploads(dataset.path()).unwrap();

        assert_eq!(sweep, LeakedUploadSweep::default());
        assert!(
            referenced.exists(),
            "transaction-referenced file must survive"
        );
    }

    #[test]
    fn test_sweep_missing_data_dir_is_a_noop() {
        let dataset = TempDir::new().unwrap();
        let sweep = sweep_leaked_uploads(&dataset.path().join("absent.lance")).unwrap();
        assert_eq!(sweep, LeakedUploadSweep::default());
    }
}
