use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};

/// A tool available to the model.
#[derive(Debug, Clone, PartialEq)]
pub enum Tool {
    /// A [function definition](https://platform.openai.com/docs/api-reference/chat/create#chat_create-tools),
    /// i.e. `{"name": .., "description": .., "parameters": ..}`. Write `#[openai_function]` above a
    /// Rust function to generate it with `function_name::openai_schema()`.
    Function(Value),
    /// Sent as it is. For provider specific tools, e.g. `json!({"type": "web_search"})`
    Custom(Value),
}
impl Tool {
    /// Name of the function, if it's a `Tool::Function`.
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Function(function) => function["name"].as_str(),
            Self::Custom(_) => None,
        }
    }
}
impl Serialize for Tool {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Function(function) => {
                json!({"type": "function", "function": function}).serialize(serializer)
            }
            Self::Custom(tool) => tool.serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        Ok(match value["type"].as_str() {
            Some("function") if value["function"].is_object() => {
                Self::Function(value["function"].take())
            }
            _ => Self::Custom(value),
        })
    }
}

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model will not call any tool.
    None,
    /// Default model behavior. Model decides whether to call tools or respond with text.
    Auto,
    /// Model must call one or more tools.
    Required,
    /// Model must call the function with this name.
    Function(String),
}
impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function(name) => {
                json!({"type": "function", "function": {"name": name}}).serialize(serializer)
            }
        }
    }
}
impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.as_str() {
            Some("none") => Ok(Self::None),
            Some("auto") => Ok(Self::Auto),
            Some("required") => Ok(Self::Required),
            _ => value["function"]["name"]
                .as_str()
                .map(|name| Self::Function(name.to_string()))
                .ok_or_else(|| serde::de::Error::custom(format!("invalid tool_choice: {value}"))),
        }
    }
}

/// Format of the reply.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    Text,
    /// Any JSON object. The prompt must ask for JSON too.
    JsonObject,
    /// JSON following `schema`. In strict mode, every object needs `"additionalProperties": false`
    /// and all its properties in `required`, which `#[openai_schema]` ensures. See
    /// [Structured outputs](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas).
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}
impl ResponseFormat {
    /// Strict JSON schema named `response`. For a Rust type, write `#[openai_schema]` above it and
    /// pass `Type::openai_schema()`.
    pub fn json_schema(schema: Value) -> Self {
        Self::JsonSchema {
            name: "response".into(),
            schema,
            strict: true,
        }
    }
}
impl Serialize for ResponseFormat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text => json!({"type": "text"}),
            Self::JsonObject => json!({"type": "json_object"}),
            Self::JsonSchema {
                name,
                schema,
                strict,
            } => json!({
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": strict}
            }),
        }
        .serialize(serializer)
    }
}

/// Value of `reasoning_effort`. Supported values depend on the model.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}
