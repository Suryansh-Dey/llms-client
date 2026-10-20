use derive_getters::{Dissolve, Getters};
use serde::{Deserialize, Serialize};

use crate::openai::types::request::{FunctionCall, Part, PartType, Role};

#[derive(Dissolve, Serialize, Deserialize, Getters, Debug, Clone)]
pub struct Chat {
    role: Role,
    parts: Vec<Part>,
}
impl Chat {
    pub(crate) const EMPTY_REPLY: Chat = Chat {
        role: Role::Assistant,
        parts: Vec::new(),
    };
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }
    pub fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }
    pub fn parts_owned(self) -> Vec<Part> {
        self.parts
    }
    ///`separator` used to concatenate all text parts. TL;DR use "\n" as separator.
    ///Don't contain thoughts
    pub fn get_text_no_think(&self, separator: impl AsRef<str>) -> String {
        Self::extract_text(&self.parts, separator, Some(false))
    }
    ///`separator` used to concatenate all text parts. TL;DR use "\n" as separator.
    pub fn get_thoughts(&self, separator: impl AsRef<str>) -> String {
        Self::extract_text(&self.parts, separator, Some(true))
    }
    pub fn extract_text_all(parts: &[Part], separator: impl AsRef<str>) -> String {
        Self::extract_text(parts, separator, None)
    }
    pub(crate) fn extract_text(
        parts: &[Part],
        separator: impl AsRef<str>,
        thought: Option<bool>,
    ) -> String {
        parts
            .iter()
            .filter_map(|part| match part.data() {
                PartType::Text(text) if thought.is_none_or(|t| t == part.is_thought()) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<&str>>()
            .join(separator.as_ref())
    }
    ///`separator` used to concatenate all text parts. TL;DR use "\n" as separator.
    ///Includes all text including thoughts
    pub fn get_text_all(&self, separator: impl AsRef<str>) -> String {
        Self::extract_text_all(self.parts(), separator)
    }
    pub fn is_thinking(&self) -> bool {
        self.parts().iter().any(|p| p.is_thought())
    }
    /// Returns the refusal message if the model refused to answer.
    pub fn get_refusal(&self) -> Option<&str> {
        self.parts().iter().find_map(|p| match p.data() {
            PartType::Refusal(refusal) => Some(refusal.as_str()),
            _ => None,
        })
    }
    pub fn has_function_call(&self) -> bool {
        self.parts()
            .iter()
            .any(|p| matches!(p.data(), PartType::FunctionCall(_)))
    }
    pub fn get_function_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.parts().iter().filter_map(|v| match v.data() {
            PartType::FunctionCall(call) => Some(call),
            _ => None,
        })
    }
}
