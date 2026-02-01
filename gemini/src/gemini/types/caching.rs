use super::request::{SystemInstruction, Tool, ToolConfig};
use derive_new::new;
use getset::Getters;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

#[derive(Serialize, Deserialize, Debug, Clone, Getters)]
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

#[derive(Error, Debug)]
pub enum CachedContentBuilderError {
    #[error("Both ttl and expire_time cannot be set")]
    ///Both ttl and expire_time cannot be set
    TtlAndExpireTime,
}

#[derive(Debug, Clone, Default)]
pub struct CachedContentBuilder {
    name: Option<String>,
    display_name: Option<String>,
    model: String,
    system_instruction: Option<SystemInstruction>,
    contents: Option<Vec<super::request::Chat>>,
    tools: Option<Vec<Tool>>,
    tool_config: Option<ToolConfig>,
    expire_time: Option<String>,
    ttl: Option<String>,
}

impl CachedContentBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            model: if model.starts_with("models/") {
                model
            } else {
                format!("models/{}", model)
            },
            ..Default::default()
        }
    }
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }
    pub fn system_instruction(mut self, system_instruction: SystemInstruction) -> Self {
        self.system_instruction = Some(system_instruction);
        self
    }
    pub fn contents(mut self, contents: Vec<super::request::Chat>) -> Self {
        self.contents = Some(contents);
        self
    }
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn tool_config(mut self, tool_config: ToolConfig) -> Self {
        self.tool_config = Some(tool_config);
        self
    }
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(format!("{}s", ttl.as_secs()));
        self
    }
    /// Expected format: RFC 3339 format e.g. '2014-10-02T15:01:23Z'
    pub fn expire_time(mut self, expire_time: impl Into<String>) -> Self {
        let expire_time = expire_time.into();
        self.expire_time = Some(expire_time);
        self
    }
    pub fn build(self) -> Result<CachedContent, CachedContentBuilderError> {
        if self.expire_time.is_some() && self.ttl.is_some() {
            return Err(CachedContentBuilderError::TtlAndExpireTime);
        }
        Ok(CachedContent {
            name: self.name,
            display_name: self.display_name,
            model: self.model,
            system_instruction: self.system_instruction,
            contents: self.contents,
            tools: self.tools,
            tool_config: self.tool_config,
            create_time: None,
            update_time: None,
            expire_time: self.expire_time,
            ttl: self.ttl,
        })
    }
}
