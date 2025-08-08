use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageDeltaUsage {
    #[serde(rename = "input_tokens")]
    pub input: u32,
    #[serde(rename = "output_tokens")]
    pub output: u32,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "cache_read_input_tokens"
    )]
    pub cache_read_input: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum ActiveState {
    #[default]
    Idle,
    Thinking {
        content_index: u32,
    },
    Text {
        content_index: u32,
    },
    Tool,
}

impl ActiveState {
    #[must_use]
    pub fn content_index(self) -> Option<u32> {
        match self {
            Self::Thinking { content_index } | Self::Text { content_index } => Some(content_index),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct StreamState {
    pub model: String,
    pub message_id: String,
    pub state: ActiveState,
    pub next_content_index: u32,
    pub usage_data: MessageDeltaUsage,
    pub tool_calls: HashMap<u32, ToolCallState>,
    pub tool_index: Option<u32>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct ToolCallState {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
    pub content_index: Option<u32>,
}
