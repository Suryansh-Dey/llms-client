use derive_new::new;
use serde::{Deserialize, Serialize};
use serde_json::{Value, from_value};
use std::collections::VecDeque;

use super::ask::GeminiResponse;

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    developer,
    model,
}

#[derive(Serialize, Deserialize, Clone, new)]
pub struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize, Deserialize, Clone, new)]
pub struct ExecutableCode {
    language: String,
    code: String,
}

#[derive(Serialize, Deserialize, Clone, new)]
pub struct CodeExecuteResult {
    outcome: String,
    output: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub enum Part {
    text(String),
    inline_data(InlineData),
    executable_code(ExecutableCode),
    code_execute_result(CodeExecuteResult),
}

#[derive(Serialize, new)]
pub struct Chat {
    role: Role,
    parts: Vec<Part>,
}

#[derive(Serialize, new)]
pub struct SystemInstruction<'a> {
    parts: &'a [Part],
}

#[derive(Serialize, new)]
pub struct GeminiBody<'a> {
    system_instruction: Option<&'a SystemInstruction<'a>>,
    contents: &'a [&'a Chat],
    generation_config: Option<&'a Value>,
}

#[derive(Serialize)]
pub struct Session {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
}
impl Session {
    pub fn new(history_limit: usize) -> Self {
        Self {
            history: VecDeque::new(),
            history_limit,
            chat_no: 0,
        }
    }
    pub fn get_history_as_vecdeque(&self) -> &VecDeque<Chat> {
        &self.history
    }
    pub(super) fn get_history_as_vecdeque_mut(&mut self) -> &mut VecDeque<Chat> {
        &mut self.history
    }
    pub fn get_chat_no(&self) -> usize {
        self.chat_no
    }
    pub fn get_history(&self) -> Vec<&Chat> {
        let (left, right) = self.history.as_slices();
        left.iter().chain(right.iter()).collect()
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    pub fn get_parts_mut(&mut self, chat_previous_no: usize) -> Option<&mut Vec<Part>> {
        let history_length = self.get_history_length();
        self.history
            .get_mut(history_length - chat_previous_no)
            .map(|chat| &mut chat.parts)
    }
    fn add_chat(&mut self, chat: Chat) -> &mut Self {
        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.history_limit {
            self.history.pop_front();
        }
        self
    }
    /// `parts` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn ask(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::user, parts))
    }
    pub fn ask_string(&mut self, prompt: String) -> &mut Self {
        self.add_chat(Chat::new(Role::user, vec![Part::text(prompt)]))
    }
    pub(super) fn update(
        &mut self,
        response: &GeminiResponse,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let history = &mut self.history;
        let reply_parts = response.get_parts_as_value();

        if let Some(chat) = history.back_mut() {
            if let Role::model = chat.role {
                concatinate_parts(
                    &mut chat.parts,
                    reply_parts.ok_or::<Box<dyn std::error::Error>>(
                        "Invalid response with no parts arrary".into(),
                    )?,
                )?;
            } else {
                history.push_back(Chat::new(
                    Role::model,
                    reply_parts
                        .ok_or("Failed to parse into parts")?
                        .iter()
                        .map(|part| from_value(part.to_owned()))
                        .collect::<Result<Vec<_>, _>>()?,
                ));
            }
        } else {
            panic!("Cannot update an empty session");
        }
        Ok(())
    }
    pub fn last_reply(&self) -> Option<&[Part]> {
        if let Some(reply) = self.get_history_as_vecdeque().back() {
            Some(&reply.parts)
        } else {
            None
        }
    }
    pub(super) fn last_reply_mut(&mut self) -> Option<&mut [Part]> {
        if let Some(reply) = self.get_history_as_vecdeque_mut().back_mut() {
            Some(reply.parts.as_mut_slice())
        } else {
            None
        }
    }
    ///`seperator` used to concatinate all text parts. TL;DR use "\n" as seperator.
    pub fn last_reply_text(&self, seperator: &str) -> Option<String> {
        let parts = self.last_reply();
        if let Some(parts) = parts {
            let mut concatinated_string = String::new();
            for part in parts {
                if let Part::text(text) = part {
                    concatinated_string.push_str(text);
                    concatinated_string.push_str(seperator);
                }
            }
            Some(concatinated_string)
        } else {
            None
        }
    }
    /// If last message is a question from user then only that is removed else the model reply and
    /// the user's question (just before model reply)
    pub fn forget_last_conversation(&mut self) {
        self.history.pop_back();
        if let Some(Chat {
            role: Role::user,
            parts: _,
        }) = self.history.back()
        {
            self.history.pop_back();
        }
    }
}
pub(super) fn concatinate_parts(
    updating: &mut Vec<Part>,
    updator: &[Value],
) -> Result<(), Box<dyn std::error::Error>> {
    for updator_part in updator {
        if let Some(updator_text) = updator_part["text"].as_str() {
            if let Some(Part::text(updating_text)) =
                updating.iter_mut().find(|e| matches!(e, Part::text(_)))
            {
                updating_text.push_str(updator_text);
                continue;
            }
        } else if let Some(updator_data) = updator_part["inline_data"]["data"].as_str() {
            if let Some(Part::inline_data(updating_data)) = updating
                .iter_mut()
                .find(|e| matches!(e, Part::inline_data(_)))
            {
                updating_data.data.push_str(updator_data);
                continue;
            }
        } else if let Some(updator_data) = updator_part["executable_code"]["code"].as_str() {
            if let Some(Part::executable_code(updating_data)) = updating
                .iter_mut()
                .find(|e| matches!(e, Part::executable_code(_)))
            {
                updating_data.code.push_str(updator_data);
                continue;
            }
        } else if let Some(updator_data) = updator_part["code_execute_result"]["output"].as_str() {
            if let Some(Part::code_execute_result(updating_data)) = updating
                .iter_mut()
                .find(|e| matches!(e, Part::code_execute_result(_)))
            {
                updating_data.output.push_str(updator_data);
                continue;
            }
        } else {
            return Err(format!("Unsupported part found: {updator_part}").into());
        }
        updating.push(from_value(updator_part.to_owned())?);
    }
    Ok(())
}
