use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    conversion::think_parser::ThinkTagParser,
    models::openai::{OpenAIStreamChoice, OpenAIStreamToolCall},
};

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
        via_think_tag: bool,
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
            Self::Thinking { content_index, .. } | Self::Text { content_index } => {
                Some(content_index)
            }
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
    pub think_parser: ThinkTagParser,
}

#[derive(Debug)]
pub enum NextState<'a> {
    Tool(&'a Vec<OpenAIStreamToolCall>),
    Finish(&'a String),
    Think { via_think_tag: bool },
    Text,
    Idle,
}

pub fn decide_next_state<'a>(
    choice: &'a OpenAIStreamChoice,
    parser: &ThinkTagParser,
) -> NextState<'a> {
    choice
        .delta
        .tool_calls
        .as_ref()
        .map(NextState::Tool)
        .or(choice.finish_reason.as_ref().map(NextState::Finish))
        .or_else(|| {
            (choice.delta.has_think_tag() && parser.is_thinking_allowed()).then_some(
                NextState::Think {
                    via_think_tag: true,
                },
            )
        })
        .or_else(|| {
            choice
                .delta
                .get_reasoning()
                .is_some()
                .then_some(NextState::Think {
                    via_think_tag: false,
                })
        })
        .or_else(|| {
            choice
                .delta
                .content
                .as_ref()
                .filter(|text| !text.is_empty())
                .map(|_| NextState::Text)
        })
        .unwrap_or(NextState::Idle)
}

pub fn decide_after_tool(choice: &OpenAIStreamChoice) -> NextState<'_> {
    choice
        .finish_reason
        .as_ref()
        .map(NextState::Finish)
        .or(choice.delta.tool_calls.as_ref().map(NextState::Tool))
        .unwrap_or(NextState::Idle)
}

pub fn decide_after_reasoning(choice: &OpenAIStreamChoice) -> NextState<'_> {
    choice
        .delta
        .tool_calls
        .as_ref()
        .map(NextState::Tool)
        .or(choice.finish_reason.as_ref().map(NextState::Finish))
        .or_else(|| {
            choice
                .delta
                .content
                .as_ref()
                .filter(|text| !text.is_empty())
                .map(|_| NextState::Text)
        })
        .or_else(|| {
            choice
                .delta
                .get_reasoning()
                .is_some()
                .then_some(NextState::Think {
                    via_think_tag: false,
                })
        })
        .unwrap_or(NextState::Idle)
}

pub fn decide_after_text(choice: &OpenAIStreamChoice) -> NextState<'_> {
    choice
        .delta
        .tool_calls
        .as_ref()
        .map(NextState::Tool)
        .or(choice.finish_reason.as_ref().map(NextState::Finish))
        .or_else(|| {
            choice
                .delta
                .content
                .as_ref()
                .filter(|text| !text.is_empty())
                .map(|_| NextState::Text)
        })
        .unwrap_or(NextState::Idle)
}

#[derive(Debug, Default, Clone)]
pub struct ToolCallState {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
    pub content_index: Option<u32>,
}
