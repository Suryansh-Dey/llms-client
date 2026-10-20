use open_client::openai::embed::OpenAIEmbedding;
use std::env;

#[tokio::main]
async fn main() {
    // 1. Create the Embedding client. Use `.set_base_url()` for other APIs.
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let embedder = OpenAIEmbedding::new(api_key, "text-embedding-3-small")
        // Optional: reduce dimension (supported by text-embedding-3 and later)
        .set_dimensions(256);

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

    // 4. Many texts in one request, one embedding per text in the same order
    let response = embedder
        .embed_texts(vec!["I love Rust".into(), "Rust loves me".into()])
        .await
        .unwrap();
    println!("Got {} embeddings", response.embeddings().len());
}
