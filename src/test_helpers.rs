/// Shared test helpers for use across all test modules.
///
/// This module is compiled only when running tests.
#[cfg(test)]
pub(crate) fn make_test_chunk(id: &str, content: &str) -> crate::HierarchicalChunk {
    crate::HierarchicalChunk {
        id: id.to_string(),
        content: content.to_string(),
        embedding: Some(vec![0.0f32; 384]),
        level: crate::chunk::ChunkLevel(1),
        parent_id: None,
        path: "test".to_string(),
        source_file: "test".to_string(),
        heading: Some(format!("Heading for {}", id)),
        start_offset: 0,
        end_offset: 0,
        cluster_memberships: vec![],
        entry_type: crate::chunk::EntryType::Raw,
        summarizes: vec![],
        perspectives: vec![],
        visibility: "normal".to_string(),
        relations: vec![],
        access_profile: crate::AccessProfile::new(),
        expires_at: None,
        impression_hint: None,
        impression_strength: 1.0,
    }
}
