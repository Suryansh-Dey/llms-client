use super::request::{ChatMessage, Role};
use super::response::{ChatGptResponse, ChatGptStreamChunk};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::mem::discriminant;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    history: VecDeque<ChatMessage>,
    history_limit: usize,
    chat_no: usize,
    remember_reply: bool,
}

impl Session {
    /// `history_limit`: Total number of chat messages (user + assistant/system/tool).
    /// new(2) will allow only 2 messages to be retained.
    pub fn new(history_limit: usize) -> Self {
        Self {
            history: VecDeque::new(),
            history_limit,
            chat_no: 0,
            remember_reply: true,
        }
    }
    pub fn set_remember_reply(mut self, remember: bool) -> Self {
        self.remember_reply = remember;
        self
    }
    pub fn get_history_limit(&self) -> usize {
        self.history_limit
    }
    pub fn get_history_as_vecdeque(&self) -> &VecDeque<ChatMessage> {
        &self.history
    }
    pub(super) fn get_history_as_vecdeque_mut(&mut self) -> &mut VecDeque<ChatMessage> {
        &mut self.history
    }
    pub fn get_chat_no(&self) -> usize {
        self.chat_no
    }
    pub fn get_history(&self) -> Vec<&ChatMessage> {
        let (left, right) = self.history.as_slices();
        left.iter().chain(right.iter()).collect()
    }
    pub fn get_history_cloned(&self) -> Vec<ChatMessage> {
        self.get_history().into_iter().cloned().collect()
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    pub fn get_last_chat(&self) -> Option<&ChatMessage> {
        self.history.back()
    }
    pub fn get_last_chat_mut(&mut self) -> Option<&mut ChatMessage> {
        self.history.back_mut()
    }
    fn add_chat(&mut self, chat: ChatMessage) -> &mut Self {
        if let Some(last_chat) = self.get_history_as_vecdeque_mut().back_mut() {
            if discriminant(last_chat.role()) == discriminant(chat.role()) {
                last_chat.append_content(chat.content());
                return self;
            }
        }
        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.get_history_limit() {
            self.history.pop_front();
        }
        self
    }
    pub fn ask(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_chat(ChatMessage::new(Role::User, content.into(), None, None))
    }
    pub fn reply(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_chat(ChatMessage::new(Role::Assistant, content.into(), None, None))
    }
    pub(crate) fn update_with_response<'a>(
        &mut self,
        response: &'a ChatGptResponse,
    ) -> Option<&'a ChatMessage> {
        if self.remember_reply {
            let message = response.message();
            self.add_chat(message.clone());
            Some(message)
        } else {
            if let Some(last) = self.history.back() {
                if let Role::User = last.role() {
                    self.history.pop_back();
                }
            }
            None
        }
    }
    pub(crate) fn update_stream(&mut self, chunk: &ChatGptStreamChunk) {
        for choice in &chunk.choices {
            if let Some(content) = choice.delta.content.as_ref() {
                if let Some(chat) = self.history.back_mut() {
                    if let Role::Assistant = chat.role() {
                        chat.append_content(content);
                        continue;
                    }
                }
                self.add_chat(ChatMessage::new(
                    Role::Assistant,
                    content.clone(),
                    None,
                    None,
                ));
            }
        }
    }
}


