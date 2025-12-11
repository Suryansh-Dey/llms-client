use serde_json::Value;

pub fn merge_json(into: &mut Value, from: &Value) {
    if let (Value::Object(into_map), Value::Object(from_map)) = (into, from) {
        for (k, v) in from_map {
            into_map.insert(k.clone(), v.clone());
        }
    }
}


