use crate::chatgpt::ask::ChatGpt;

#[actix_web::test]
async fn it_works() {
    let data = ChatGpt::new(
        &std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        "gpt-4o-mini",
    )
    .ask_string("Hi")
    .await
    .unwrap();
    println!("{data}");
}
