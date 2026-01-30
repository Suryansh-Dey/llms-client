use derive_new::new;
use getset::Getters;
use serde::{Deserialize, Serialize};

use super::request::{SystemInstruction, Tool, ToolConfig};

#[derive(Serialize, Deserialize, Debug, Clone, Getters, new)]
#[serde(rename_all = "camelCase")]
pub struct CachedContent {
    /// The resource name referring to the cached content. Format: `cachedContents/{id}`
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    name: Option<String>,
    /// The display name of the cached content.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    display_name: Option<String>,
    /// The name of the model to use the cached content with.
    #[get = "pub"]
    model: String,
    /// System instruction to be cached.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    system_instruction: Option<SystemInstruction>,
    /// The user's content to cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    contents: Option<Vec<super::request::Chat>>,
    /// A list of tools to be cached.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    tools: Option<Vec<Tool>>,
    /// Tool config to be cached.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    tool_config: Option<ToolConfig>,
    /// The creation time of the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    create_time: Option<String>,
    /// The update time of the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    update_time: Option<String>,
    /// The expiration time of the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    expire_time: Option<String>,
    /// The TTL (Time To Live) of the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    ttl: Option<String>, // Can be formatted as a duration string e.g. "300s"
}

#[derive(Serialize, Deserialize, Debug, Clone, Getters)]
#[serde(rename_all = "camelCase")]
pub struct CachedContentList {
    #[get = "pub"]
    cached_contents: Option<Vec<CachedContent>>,
    #[get = "pub"]
    next_page_token: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, new)]
#[serde(rename_all = "camelCase")]
pub struct CachedContentUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expire_time: Option<String>,
}
