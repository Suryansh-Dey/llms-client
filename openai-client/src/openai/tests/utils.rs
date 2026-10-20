//! `MarkdownToParts` with local files. Downloading is tested in `http.rs`.
use crate::openai::types::request::{Part, PartType};
use crate::openai::utils::MarkdownToParts;
use regex::Regex;

const IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");

/// Text parts as they are, files as `<mime type>`.
fn describe(parts: &[Part]) -> Vec<String> {
    parts
        .iter()
        .map(|part| match part.data() {
            PartType::Text(text) => text.clone(),
            PartType::InlineData(data) => format!("<{}>", data.mime_type()),
            other => panic!("Unexpected part {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn files_are_split_from_text_in_order() {
    let markdown = format!("A ![first]({IMAGE}) B ![second]({IMAGE}) C");
    let parts = MarkdownToParts::builder()
        .guess_mime_type(|url| match url.ends_with(".png") {
            true => mime::IMAGE_PNG,
            false => mime::APPLICATION_OCTET_STREAM,
        })
        .build(&markdown)
        .await
        .process();
    assert_eq!(
        describe(&parts),
        [
            format!("A ![first]({IMAGE})"),
            "<image/png>".into(),
            format!(" B ![second]({IMAGE})"),
            "<image/png>".into(),
            " C".into()
        ]
    );
}

#[tokio::test]
async fn missing_file_stays_text() {
    let markdown = format!("![gone](does/not/exist.png) and ![found]({IMAGE})");
    let parts = MarkdownToParts::builder().build(&markdown).await.process();
    assert_eq!(
        describe(&parts),
        [
            format!("![gone](does/not/exist.png) and ![found]({IMAGE})"),
            "<image/png>".into()
        ]
    );
}

#[tokio::test]
async fn custom_regex() {
    let markdown = format!("See [[{IMAGE}]]");
    let parts = MarkdownToParts::builder()
        .regex(Regex::new(r"\[\[(.*?)\]\]").unwrap())
        .build(&markdown)
        .await
        .process();
    assert_eq!(
        describe(&parts),
        [format!("See [[{IMAGE}]]"), "<image/png>".into()]
    );
}

#[tokio::test]
async fn no_files() {
    let parts = MarkdownToParts::builder()
        .build("Just text")
        .await
        .process();
    assert_eq!(describe(&parts), ["Just text"]);
}
