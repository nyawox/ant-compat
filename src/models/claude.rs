use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeContent {
    Text(String),
    Array(Vec<ClaudeContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub source: Option<ImageSource>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<Value>,
    pub tool_use_id: Option<String>,
    pub content: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: ClaudeContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeSystem {
    Text(String),
    Array(Vec<ClaudeContentBlock>),
}

impl ClaudeMessagesRequest {
    #[must_use]
    pub fn find_tool_name_by_id(&self, tool_use_id: &str) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .filter_map(|message| {
                if let ("assistant", ClaudeContent::Array(blocks)) =
                    (&message.role[..], &message.content)
                {
                    Some(blocks.iter())
                } else {
                    None
                }
            })
            .flatten()
            .find_map(|block| {
                if block.block_type == "tool_use" && block.id.as_deref() == Some(tool_use_id) {
                    block.name.clone()
                } else {
                    None
                }
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessagesRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    pub system: Option<ClaudeSystem>,
    pub max_tokens: u32,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<ClaudeTool>>,
    pub tool_choice: Option<ClaudeToolChoice>,
    pub thinking: Option<ClaudeThinking>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart(MessageStart),
    ContentBlockStart(ContentBlockStart),
    ContentBlockDelta(ContentBlockDelta),
    ContentBlockStop(ContentBlockStop),
    MessageDelta(MessageDelta),
    MessageStop(MessageStop),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStart {
    pub message: ClaudeStreamMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: ClaudeStreamUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeStreamUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStart {
    pub index: u32,
    pub content_block: ContentBlock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockDelta {
    pub index: u32,
    pub delta: Delta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlockStop {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaInfo {
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    pub delta: MessageDeltaInfo,
    pub usage: super::shared::MessageDeltaUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStop {}

impl AnthropicStreamEvent {
    #[must_use]
    pub fn to_parts(&self) -> (&'static str, Value) {
        match self {
            AnthropicStreamEvent::MessageStart(_) => (
                "message_start",
                serde_json::to_value(self).unwrap_or_default(),
            ),
            AnthropicStreamEvent::ContentBlockStart(_) => (
                "content_block_start",
                serde_json::to_value(self).unwrap_or_default(),
            ),
            AnthropicStreamEvent::ContentBlockDelta(_) => (
                "content_block_delta",
                serde_json::to_value(self).unwrap_or_default(),
            ),
            AnthropicStreamEvent::ContentBlockStop(_) => (
                "content_block_stop",
                serde_json::to_value(self).unwrap_or_default(),
            ),
            AnthropicStreamEvent::MessageDelta(_) => (
                "message_delta",
                serde_json::to_value(self).unwrap_or_default(),
            ),
            AnthropicStreamEvent::MessageStop(_) => (
                "message_stop",
                serde_json::to_value(self).unwrap_or_default(),
            ),
        }
    }
}

pub struct FinishReason<'a>(pub Option<&'a str>);

impl FinishReason<'_> {
    #[must_use]
    pub fn to_anthropic_stop_reason(&self) -> &'static str {
        match self.0 {
            Some("length") => "max_tokens",
            Some("tool_calls") => "tool_use",
            Some("content_filter") => "stop_sequence",
            _ => "end_turn",
        }
    }
}
