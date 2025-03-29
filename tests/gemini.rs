use llms_client::gemini::ask::Gemini;
use serde_json::json;

#[actix_web::test]
async fn ask_string() {
    let response = Gemini::new(
        &std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        None,
    )
    .ask_string("Hi")
    .await
    .unwrap();
    println!("{}", Gemini::get_response_string(&response));
}

#[actix_web::test]
async fn ask_string_for_json() {
    let response = Gemini::new(
        &std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        None,
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
    .ask_string("Calssify these words: 
   [\"Joy\", \"Success\", \"Love\", \"Hope\", \"Confidence\", \"Peace\", \"Victory\", \"Harmony\", \"Inspiration\", \"Gratitude\", \"Prosperity\", \"Strength\", \"Freedom\", \"Comfort\", \"Brilliance\" \"Fear\", \"Failure\", \"Hate\", \"Doubt\", \"Pain\", \"Suffering\", \"Loss\", \"Anxiety\", \"Despair\", \"Betrayal\", \"Weakness\", \"Chaos\", \"Misery\", \"Frustration\", \"Darkness\"]
")
    .await
    .unwrap();
    println!("{:?}", Gemini::get_response_json(&response).unwrap());
}
