use super::types::*;
use awc::Client;
use serde_json::{Value, json};
use std::time::Duration;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct Gemini<'a> {
    client: Client,
    api_key: &'a str,
    model: &'a str,
    sys_prompt: Option<&'a [Part<'a>]>,
    generation_config: Option<Value>,
}
impl<'a> Gemini<'a> {
    pub fn new(api_key: &'a str, model: &'a str, sys_prompt: Option<&'a [Part<'a>]>) -> Gemini<'a> {
        Self {
            client: Client::builder().timeout(Duration::from_secs(30)).finish(),
            api_key,
            model,
            sys_prompt,
            generation_config: None,
        }
    }
    pub fn set_generation_config(&mut self, generation_config: Value) -> &mut Self {
        self.generation_config = Some(generation_config);
        self
    }
    pub fn set_json_mode(&mut self, schema: Value) -> &Self {
        if let None = self.generation_config {
            self.generation_config = Some(json!({
                "response_mime_type": "application/json",
                "response_schema":schema
            }))
        } else if let Some(config) = self.generation_config.as_mut() {
            config["response_mime_type"] = "application/json".into();
            config["response_schema"] = schema.into();
        }
        self
    }

    pub async fn ask_string(&self, question: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(
                &[Chat::new(Role::user, &[Part::text(question)])],
                self.sys_prompt,
                self.generation_config.as_ref(),
            ))
            .await?
            .json()
            .await?;

        Ok(response)
    }
    pub fn get_response_string(response: &Value) -> String {
        response["candidates"][0]["content"]["parts"][0]["text"].to_string()
    }
    pub fn get_response_json(response: &Value) -> Result<Value, serde_json::Error> {
        let string = response["candidates"][0]["content"]["parts"][0]["text"].to_string();
        let unescaped_str = string.replace("\\\"", "\"");
        serde_json::from_str(&unescaped_str[1..unescaped_str.len() - 1])
    }
}

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
