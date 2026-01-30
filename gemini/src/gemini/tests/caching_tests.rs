use crate::gemini::types::caching::CachedContent;
use serde_json::json;

#[test]
fn test_cached_content_serialization() {
    let cached_content = CachedContent::new(
        None, // name
        None, // display_name
        "models/gemini-1.5-flash-001".to_string(), // model
        None, // system_instruction
        Some(vec![]), // contents
        Some(vec![]), // tools
        None, // tool_config
        None, // create_time
        None, // update_time
        None, // expire_time
        None, // ttl
    );
    let json = serde_json::to_value(&cached_content).unwrap();
    assert_eq!(json["model"], "models/gemini-1.5-flash-001");
}

#[test]
fn test_cached_content_deserialization() {
    let json_data = json!({
        "name": "cachedContents/12345",
        "displayName": "Test Cache",
        "model": "models/gemini-1.5-pro-001",
        "createTime": "2024-01-01T00:00:00Z",
        "updateTime": "2024-01-01T00:00:00Z",
        "expireTime": "2024-01-02T00:00:00Z",
        "ttl": "86400s"
    });

    let cached_content: CachedContent = serde_json::from_value(json_data).unwrap();
    assert_eq!(cached_content.name().as_ref().unwrap(), "cachedContents/12345");
    assert_eq!(cached_content.display_name().as_ref().unwrap(), "Test Cache");
    assert_eq!(cached_content.model(), "models/gemini-1.5-pro-001");
    assert_eq!(cached_content.ttl().as_ref().unwrap(), "86400s");
}
