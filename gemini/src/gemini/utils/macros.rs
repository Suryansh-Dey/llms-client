use serde_json::{json, Value};

/// Trait for types that can generate a Gemini-compatible JSON schema.
pub trait GeminiSchema {
    fn gemini_schema() -> Value;
}

macro_rules! impl_primitive {
    ($ty:ty, $schema_type:expr) => {
        impl GeminiSchema for $ty {
            fn gemini_schema() -> Value {
                json!({ "type": $schema_type })
            }
        }
    };
}

impl_primitive!(String, "STRING");
impl_primitive!(bool, "BOOLEAN");
impl_primitive!(f32, "NUMBER");
impl_primitive!(f64, "NUMBER");
impl_primitive!(i8, "INTEGER");
impl_primitive!(i16, "INTEGER");
impl_primitive!(i32, "INTEGER");
impl_primitive!(i64, "INTEGER");
impl_primitive!(i128, "INTEGER");
impl_primitive!(isize, "INTEGER");
impl_primitive!(u8, "INTEGER");
impl_primitive!(u16, "INTEGER");
impl_primitive!(u32, "INTEGER");
impl_primitive!(u64, "INTEGER");
impl_primitive!(u128, "INTEGER");
impl_primitive!(usize, "INTEGER");

impl GeminiSchema for &str {
    fn gemini_schema() -> Value {
        json!({ "type": "STRING" })
    }
}

impl<T: GeminiSchema> GeminiSchema for Vec<T> {
    fn gemini_schema() -> Value {
        json!({
            "type": "ARRAY",
            "items": T::gemini_schema()
        })
    }
}

impl<T: GeminiSchema> GeminiSchema for Option<T> {
    fn gemini_schema() -> Value {
        let mut schema = T::gemini_schema();
        if let Some(obj) = schema.as_object_mut() {
            obj.insert("nullable".to_string(), json!(true));
        }
        schema
    }
}
