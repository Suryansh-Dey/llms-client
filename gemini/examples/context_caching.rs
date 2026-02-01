use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::caching::CachedContentBuilder;
use gemini_client_api::gemini::types::request::InlineData;
use gemini_client_api::gemini::types::sessions::Session;
use std::env;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);
    let mut session = Session::new(10);

    session.ask("What is there in this pdf".repeat(200)); //Faking big context for example
    session.ask(InlineData::from_url("https://bitmesra.ac.in/UploadedDocuments/admingo/files/221225_List%20of%20Holiday_2026_26.pdf").await.unwrap());

    let cached_content_req = CachedContentBuilder::new("gemini-2.5-flash")
        .display_name("Simulated Large Doc")
        .contents(
            session
                .get_history()
                .into_iter()
                .map(|e| e.to_owned())
                .collect(),
        )
        .ttl(Duration::from_secs(300))
        .build()
        .unwrap();

    println!("Creating cache...");
    match ai.create_cache(&cached_content_req).await {
        Ok(cache) => {
            println!("Cache created: {}", cache.name().as_ref().unwrap());

            // 2. Use the cache in a request
            let mut session = Session::new(10);
            let prompt = "Summarize the cached document.";
            println!("User: {}", prompt);

            // Create a new client instance that uses the cache
            let ai_with_cache = ai
                .clone()
                .set_cached_content(cache.name().as_ref().unwrap());

            match ai_with_cache.ask(session.ask(prompt)).await {
                Ok(response) => {
                    println!("\nGemini: {}", response.get_chat().get_text_no_think(""));
                }
                Err(e) => eprintln!("Error asking Gemini: {:?}", e),
            }

            // 3. List caches
            println!("\nListing caches...");
            match ai.list_caches().await {
                Ok(list) => {
                    if let Some(caches) = list.cached_contents() {
                        for c in caches {
                            println!("- {}", c.name().as_ref().unwrap_or(&"Unknown".to_string()));
                        }
                    } else {
                        println!("No caches found.");
                    }
                }
                Err(e) => eprintln!("Error listing caches: {:?}", e),
            }

            // 4. Delete the cache
            println!("\nDeleting cache...");
            match ai.delete_cache(cache.name().as_ref().unwrap()).await {
                Ok(_) => println!("Cache deleted."),
                Err(e) => eprintln!("Error deleting cache: {:?}", e),
            }
        }
        Err(e) => {
            eprintln!("Failed to create cache: {:?}", e);
        }
    }
}
