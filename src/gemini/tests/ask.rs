use crate::gemini::ask::Gemini;
use crate::gemini::types::{Part, SystemInstruction};
use serde_json::json;

#[actix_web::test]
async fn ask_string() {
    let response = Gemini::new(
        &std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        None,
    )
    .ask_string("Hi".to_string())
    .await
    .unwrap();
    println!("{}", response.get_as_string().unwrap());
}

#[actix_web::test]
async fn ask_string_for_json() {
    let response = Gemini::new(
        &std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        Some(SystemInstruction::new(&[Part::text("Calssify the given words".to_string())])),
    )
    .set_json_mode(json!({
        "type": "object",
        "properties": {
            "positive":{
                "type":"array",
                "items":{"type":"string"}
            },
            "negetive":{
                "type":"array",
                "items":{"type":"string"}
            }
        }
    }))
    .ask_string("[\"Joy\", \"Success\", \"Love\", \"Hope\", \"Confidence\", \"Peace\", \"Victory\", \"Harmony\", \"Inspiration\", \"Gratitude\", \"Prosperity\", \"Strength\", \"Freedom\", \"Comfort\", \"Brilliance\" \"Fear\", \"Failure\", \"Hate\", \"Doubt\", \"Pain\", \"Suffering\", \"Loss\", \"Anxiety\", \"Despair\", \"Betrayal\", \"Weakness\", \"Chaos\", \"Misery\", \"Frustration\", \"Darkness\"]
".to_string())
    .await
    .unwrap();
    println!("{:?}", response.get_as_json().unwrap());
}
