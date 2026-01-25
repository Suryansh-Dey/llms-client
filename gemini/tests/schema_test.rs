use gemini_client_api::gemini::types::request::Tool;
use gemini_client_api::gemini::{
    ask::Gemini,
    types::sessions::Session,
    utils::{GeminiSchema, gemini_function, gemini_schema},
};
use serde::Deserialize;
use serde_json::json;
use std::fmt::Debug;

#[allow(dead_code)]
#[gemini_schema]
/// A priority level
enum Priority {
    Low,
    Medium,
    High,
}

#[allow(dead_code)]
#[gemini_schema]
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
}

#[test]
fn gemini_schema_generation_test() {
    let schema = Task::gemini_schema();

    let expected = json!({
        "type": "OBJECT",
        "description": "A task to be performed",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "The title of the task"
            },
            "description": {
                "type": "STRING",
                "description": "Detailed description",
                "nullable": true
            },
            "priority": {
                "type": "STRING",
                "description": "How important it is",
                "enum": ["Low", "Medium", "High"]
            },
            "subtasks": {
                "type": "ARRAY",
                "description": "Subtasks",
                "items": {
                    "type": "STRING"
                }
            }
        },
        "required": ["title", "priority", "subtasks"]
    });

    assert_eq!(schema, expected);
}

#[allow(dead_code)]
#[gemini_schema]
struct SubTask {
    name: String,
    done: bool,
}

#[allow(dead_code)]
#[gemini_schema]
struct ComplexTask {
    title: String,
    subtasks: Vec<SubTask>,
}

#[test]
fn complex_schema_test() {
    let schema = ComplexTask::gemini_schema();
    let expected = json!({
        "type": "OBJECT",
        "properties": {
            "title": { "type": "STRING" },
            "subtasks": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": { "type": "STRING" },
                        "done": { "type": "BOOLEAN" }
                    },
                    "required": ["name", "done"]
                }
            }
        },
        "required": ["title", "subtasks"]
    });
    assert_eq!(schema, expected);
}

#[tokio::test]
async fn ask_string_for_json_with_struct() {
    #[derive(Debug, Deserialize)]
    #[gemini_schema]
    struct Schema {
        positive: Vec<String>,
        negative: Vec<String>,
    }
    let mut session = Session::new(6).set_remember_reply(false);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        Some("Classify the given words".into()),
    )
    .set_json_mode(Schema::gemini_schema())
    .ask(session.ask_string(r#"["Joy", "Success", "Love", "Hope", "Confidence", "Peace", "Victory", "Harmony", "Inspiration", "Gratitude", "Prosperity", "Strength", "Freedom", "Comfort", "Brilliance" "Fear", "Failure", "Hate", "Doubt", "Pain", "Suffering", "Loss", "Anxiety", "Despair", "Betrayal", "Weakness", "Chaos", "Misery", "Frustration", "Darkness"]"#))
    .await
    .unwrap();

    let json: Schema = response.get_json().unwrap();
    println!(
        "positives:{:?}\nnegatives:{:?}",
        json.positive, json.negative
    )
}

#[allow(dead_code)]
#[gemini_function]
/// Get the current weather in a given location
fn get_weather(
    /// The city and state, e.g. San Francisco, CA
    location: String,
    /// The temperature unit to use
    unit: Option<String>,
) {
    let _ = (location, unit);
}

#[test]
fn gemini_function_schema_test() {
    let schema = get_weather::gemini_schema();
    let expected = json!({
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "STRING",
                    "description": "The temperature unit to use",
                    "nullable": true
                }
            },
            "required": ["location"]
        }
    });

    assert_eq!(schema, expected);

    let tool = Tool::functionDeclarations(vec![schema]);
    if let Tool::functionDeclarations(decls) = tool {
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "get_weather");
    } else {
        panic!("Expected Tool::functionDeclarations");
    }
}
