use open_client::openai::types::request::Tool;
use open_client::openai::{
    types::sessions::Session,
    utils::{OpenAISchema, openai_function, openai_schema},
};
use serde::Deserialize;
use serde_json::json;
use std::fmt::Debug;

#[allow(dead_code)]
#[openai_schema]
/// A priority level
enum Priority {
    Low,
    Medium,
    High,
}

#[allow(dead_code)]
#[openai_schema]
/// A task to be performed
struct Task {
    /// The title of the task
    title: String,
    /// Detailed description
    description: Option<String>,
    /// How important it is
    priority: Priority,
    /// Subtasks
    subtasks: Vec<String>,
    /// Who owns it
    owner: Option<Priority>,
}

#[test]
fn openai_schema_generation_test() {
    let schema = Task::openai_schema();

    let expected = json!({
        "type": "object",
        "description": "A task to be performed",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the task"
            },
            "description": {
                "type": ["string", "null"],
                "description": "Detailed description"
            },
            "priority": {
                "type": "string",
                "description": "How important it is",
                "enum": ["Low", "Medium", "High"]
            },
            "subtasks": {
                "type": "array",
                "description": "Subtasks",
                "items": {
                    "type": "string"
                }
            },
            "owner": {
                "anyOf": [
                    {"type": "string", "description": "A priority level", "enum": ["Low", "Medium", "High"]},
                    {"type": "null"}
                ],
                "description": "Who owns it"
            }
        },
        "required": ["title", "description", "priority", "subtasks", "owner"],
        "additionalProperties": false
    });

    assert_eq!(schema, expected);
}

#[allow(dead_code)]
#[openai_schema]
struct SubTask {
    name: String,
    done: bool,
}

#[allow(dead_code)]
#[openai_schema]
struct ComplexTask {
    title: String,
    subtasks: Vec<SubTask>,
}

#[test]
fn complex_schema_test() {
    let schema = ComplexTask::openai_schema();
    let expected = json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "done": { "type": "boolean" }
                    },
                    "required": ["name", "done"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["title", "subtasks"],
        "additionalProperties": false
    });
    assert_eq!(schema, expected);
}

mod live {
    use super::*;
    use open_client::openai::ask::{DEFAULT_BASE_URL, OpenAI};

    #[tokio::test]
    async fn ask_string_for_json_with_struct() {
        #[derive(Debug, Deserialize)]
        #[openai_schema]
        struct Schema {
            positive: Vec<String>,
            negative: Vec<String>,
        }
        let mut session = Session::new(6).set_remember_reply(false);
        let response = OpenAI::new(
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
            std::env::var("OPENAI_MODEL").unwrap_or("gpt-5-mini".into()),
            Some("Classify the given words".into()),
        )
        .set_base_url(std::env::var("OPENAI_BASE_URL").unwrap_or(DEFAULT_BASE_URL.into()))
        .set_json_mode(Schema::openai_schema())
        .ask(session.ask(r#"["Joy", "Success", "Love", "Hope", "Confidence", "Peace", "Victory", "Harmony", "Inspiration", "Gratitude", "Prosperity", "Strength", "Freedom", "Comfort", "Brilliance" "Fear", "Failure", "Hate", "Doubt", "Pain", "Suffering", "Loss", "Anxiety", "Despair", "Betrayal", "Weakness", "Chaos", "Misery", "Frustration", "Darkness"]"#))
        .await
        .unwrap();

        let json: Schema = response.get_json().unwrap();
        println!(
            "positives:{:?}\nnegatives:{:?}",
            json.positive, json.negative
        )
    }
}

#[allow(dead_code)]
#[openai_function]
/// Get the current weather in a given location
fn get_weather(
    /// The city and state, e.g. San Francisco, CA
    location: String,
    /// The temperature unit to use
    unit: Option<String>,
) {
    let _ = (location, unit);
}

#[allow(dead_code)]
#[openai_function]
fn no_doc() {}

#[test]
fn openai_function_schema_test() {
    let schema = get_weather::openai_schema();
    assert_eq!(get_weather::name(&schema), Some("get_weather"));
    let expected = json!({
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": ["string", "null"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location", "unit"],
            "additionalProperties": false
        }
    });

    assert_eq!(schema, expected);
    // Optional arguments can be null or missing
    assert_eq!(
        get_weather::parse_arguments(&json!({"location": "Paris", "unit": null})).unwrap(),
        ("Paris".to_string(), None)
    );
    assert_eq!(
        get_weather::parse_arguments(&json!({"location": "Paris"})).unwrap(),
        ("Paris".to_string(), None)
    );

    let tool = Tool::Function(schema);
    assert_eq!(tool.name(), Some("get_weather"));
    assert_eq!(
        serde_json::to_value(&tool).unwrap()["function"]["name"],
        "get_weather"
    );

    // No description when there are no doc comments
    assert!(no_doc::openai_schema().get("description").is_none());
}
