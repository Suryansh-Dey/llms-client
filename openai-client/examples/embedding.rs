//! Semantic search: the documents most similar in meaning to a query, using embeddings.
use open_client::embeddings::Embedder;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut embedder = Embedder::from_env("text-embedding-3-small")?;
    // Optional: smaller embeddings, supported by text-embedding-3 and later.
    embedder.dimensions(256);

    let documents = [
        "Rust guarantees memory safety without a garbage collector.",
        "Tokio is an asynchronous runtime for Rust.",
        "The Eiffel Tower was finished in 1889.",
        "Photosynthesis turns light into chemical energy.",
    ];
    let query = "How do I write async code in Rust?";

    // One request for all texts. Embeddings are in the same order as the texts.
    let mut texts = documents.to_vec();
    texts.push(query);
    let mut embeddings = embedder.embed(texts).await?.into_embeddings();
    let query_embedding = embeddings.pop().unwrap();

    let mut ranked: Vec<(f32, &str)> = embeddings
        .iter()
        .zip(documents)
        .map(|(embedding, document)| (cosine_similarity(embedding, &query_embedding), document))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!("Query: {query}");
    for (similarity, document) in ranked {
        println!("{similarity:.3}  {document}");
    }
    Ok(())
}
