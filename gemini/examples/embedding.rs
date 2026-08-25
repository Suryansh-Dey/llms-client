use gemini_client_api::gemini::embed::GeminiEmbedding;
use gemini_client_api::gemini::types::embedding::TaskType;
use std::env;

#[tokio::main]
async fn main() {
    // 1. Create the Gemini Embedding client
    // Get your API key from https://aistudio.google.com/app/apikey
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let embedder = GeminiEmbedding::new(api_key, "gemini-embedding-001")
        .set_task_type(TaskType::RetrievalDocument)
        // Optional: reduce dimension for Matryoshka Representation Learning
        .set_output_dimensionality(256);

    // 2. Generate embedding for a single text
    let prompt = "Rust is a blazing fast and memory-efficient systems programming language.";
    let response = embedder.embed_text(prompt).await.unwrap();

    // 3. Print the embedding information
    let embedding = response.embedding();
    println!("Embedding generated for: {:?}", prompt);
    println!("Total Dimensions: {}", embedding.dimension());
    println!(
        "First 5 values: {:?}",
        &embedding.values()[..usize::min(embedding.dimension(), 5)]
    );
}
