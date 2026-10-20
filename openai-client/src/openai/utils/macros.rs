use serde_json::{Value, json};

/// Trait for types that can generate an OpenAI compatible JSON schema.
///
/// Schemas are valid for [strict mode](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas):
/// all properties are `required` and `Option<T>` is nullable instead of optional.
pub trait OpenAISchema {
    fn openai_schema() -> Value;
}

macro_rules! impl_primitive {
    ($ty:ty, $schema_type:expr) => {
        impl OpenAISchema for $ty {
            fn openai_schema() -> Value {
                json!({ "type": $schema_type })
            }
        }
    };
}

impl_primitive!(String, "string");
impl_primitive!(bool, "boolean");
impl_primitive!(f32, "number");
impl_primitive!(f64, "number");
impl_primitive!(i8, "integer");
impl_primitive!(i16, "integer");
impl_primitive!(i32, "integer");
impl_primitive!(i64, "integer");
impl_primitive!(i128, "integer");
impl_primitive!(isize, "integer");
impl_primitive!(u8, "integer");
impl_primitive!(u16, "integer");
impl_primitive!(u32, "integer");
impl_primitive!(u64, "integer");
impl_primitive!(u128, "integer");
impl_primitive!(usize, "integer");

impl OpenAISchema for &str {
    fn openai_schema() -> Value {
        json!({ "type": "string" })
    }
}

impl<T: OpenAISchema> OpenAISchema for Vec<T> {
    fn openai_schema() -> Value {
        json!({
            "type": "array",
            "items": T::openai_schema()
        })
    }
}

/// Primitive types become `{"type": [type, "null"]}`, everything else `{"anyOf": [schema, {"type": "null"}]}`.
impl<T: OpenAISchema> OpenAISchema for Option<T> {
    fn openai_schema() -> Value {
        let mut schema = T::openai_schema();
        let primitive = schema.get("enum").is_none()
            && matches!(
                schema["type"].as_str(),
                Some("string" | "number" | "integer" | "boolean")
            );
        if primitive {
            schema["type"] = json!([schema["type"], "null"]);
            schema
        } else {
            json!({ "anyOf": [schema, { "type": "null" }] })
        }
    }
}
