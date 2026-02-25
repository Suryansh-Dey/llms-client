use derive_new::new;
use getset::Getters;
use serde::{Deserialize, Serialize};

use crate::gemini::types::request::{FunctionCall, Part, PartType, Role};

#[derive(Serialize, Deserialize, new, Getters, Debug, Clone)]
pub struct Chat {
    #[get = "pub"]
    role: Role,
    #[get = "pub"]
    parts: Vec<Part>,
}
impl Chat {
    pub fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    ///Don't contain thoughts
    pub fn get_text_no_think(&self, seperator: impl AsRef<str>) -> String {
        let parts = self.parts();
        let final_text = parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text) = part.data() {
                    if !part.is_thought() {
                        Some(text.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref());

        final_text
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    pub fn get_thoughts(&self, seperator: impl AsRef<str>) -> String {
        let parts = self.parts();
        let thoughts = parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text) = part.data() {
                    if part.is_thought() {
                        Some(text.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref());

        thoughts
    }
    pub fn extract_text_all(parts: &[Part], seperator: impl AsRef<str>) -> String {
        parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text_part) = part.data() {
                    // Just return the text, without checking the `thought` flag
                    Some(text_part.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref())
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    ///Includes all text including thoughts
    pub fn get_text_all(&self, seperator: impl AsRef<str>) -> String {
        Self::extract_text_all(&self.parts(), seperator)
    }
    pub fn is_thinking(&self) -> bool {
        self.parts().iter().any(|p| p.is_thought())
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
