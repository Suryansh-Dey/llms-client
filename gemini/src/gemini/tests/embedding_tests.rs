use crate::gemini::{
    embed::GeminiEmbedding,
    types::embedding::{Content, EmbedContentRequest, TaskType},
};
use std::env;

#[test]
fn test_embed_content_request_serialization() {
    let request = EmbedContentRequest {
        model: "models/text-embedding-004".to_string(),
        content: Content::from("Hello world"),
        task_type: Some(TaskType::RetrievalDocument),
        output_dimensionality: Some(256),
    };

    let json = serde_json::to_value(&request).unwrap();

    // Verify taskType and outputDimensionality are at the top level
    assert_eq!(json["model"], "models/text-embedding-004");
    assert_eq!(json["content"]["parts"][0]["text"], "Hello world");
    assert_eq!(json["taskType"], "RETRIEVAL_DOCUMENT");
    assert_eq!(json["outputDimensionality"], 256);

    // Ensure embedContentConfig does NOT exist to prevent the bug
    assert!(json.get("embedContentConfig").is_none());
}

#[test]
fn test_embed_content_request_serialization_no_options() {
    let request = EmbedContentRequest {
        model: "models/text-embedding-004".to_string(),
        content: Content::from("Hello world"),
        task_type: None,
        output_dimensionality: None,
    };

    let json = serde_json::to_value(&request).unwrap();

    assert_eq!(json["model"], "models/text-embedding-004");
    assert_eq!(json["content"]["parts"][0]["text"], "Hello world");
    assert!(json.get("taskType").is_none());
    assert!(json.get("outputDimensionality").is_none());
    assert!(json.get("embedContentConfig").is_none());
}

fn magnitude(a: Vec<f32>) -> f32 {
    let mut sum = 0.0;
    for v in a {
        sum += v * v;
    }
    return f32::sqrt(sum);
}
fn cosin_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    let mut dot = 0.0;
    assert!(a.len() == b.len());
    for (ai, bi) in a.iter().zip(&b) {
        dot += ai * bi;
    }
    return dot / magnitude(a) / magnitude(b);
}
#[tokio::test]
async fn embdding_dimension_and_similarity() {
    let embedder = GeminiEmbedding::new(
        env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set"),
        "gemini-embedding-001",
    )
    .set_task_type(TaskType::RetrievalDocument)
    .set_output_dimensionality(256);
    let prompt1 = "Rust is a blazing fast and memory-efficient systems programming language.";
    let embedding1 = embedder
        .embed_text(prompt1)
        .await
        .unwrap()
        .embedding_owned();
    assert!(embedding1.dimension() == 256);
    let prompt2 = "Rust is memory-efficient and very fast systems programming language.";
    let embedding2 = embedder
        .embed_text(prompt2)
        .await
        .unwrap()
        .embedding_owned();
    let similarity = cosin_similarity(embedding1.values_owned(), embedding2.values_owned());
    assert!(similarity > 0.98);
}
