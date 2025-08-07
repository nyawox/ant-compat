use crate::{
    conversion::request::Request,
    models::{
        claude::{ClaudeTool, ClaudeToolChoice},
        openai::{OpenAIMessage, OpenAIStreamChunk},
    },
};
use futures_util::stream::Stream;
use serde_json::Value;
use std::pin::Pin;

pub trait Adapter: Send + Sync {
    fn adapt_tools(
        &self,
        tools: Option<Vec<ClaudeTool>>,
        _request: &Request,
    ) -> Option<Vec<ClaudeTool>> {
        tools
    }

    fn adapt_tool_choice(
        &self,
        tool_choice: Option<ClaudeToolChoice>,
        _request: &Request,
    ) -> Option<ClaudeToolChoice> {
        tool_choice
    }

    fn adapt_tool_schema(&self, schema: &Value, _request: &Request) -> Value {
        schema.clone()
    }

    fn adapt_tool_description(&self, description: &str, _request: &Request) -> String {
        description.to_string()
    }

    fn adapt_system_prompt(&self, system_prompt: &str, _request: &Request) -> String {
        system_prompt.to_string()
    }

    fn adapt_user_prompt(&self, user_prompt: &str, _request: &Request) -> String {
        user_prompt.to_string()
    }

    fn adapt_temperature(&self, temperature: Option<f32>, _request: &Request) -> Option<f32> {
        temperature
    }

    fn adapt_top_p(&self, top_p: Option<f32>, _request: &Request) -> Option<f32> {
        top_p
    }

    fn adapt_max_tokens(&self, max_tokens: u32, _request: &Request) -> u32 {
        max_tokens
    }

    fn adapt_tool_result(&self, _tool_name: &str, tool_result: &str, _request: &Request) -> String {
        tool_result.to_string()
    }

    fn adapt_model(&self, model: &str, _request: &Request) -> String {
        model.to_string()
    }

    fn adapt_messages(
        &self,
        messages: Vec<OpenAIMessage>,
        _request: &Request,
    ) -> Vec<OpenAIMessage> {
        messages
    }

    fn adapt_non_stream_response(&self, response: Value, _request: &Request) -> Value {
        response
    }

    fn adapt_chunk_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
        _request: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
        stream
    }
}
