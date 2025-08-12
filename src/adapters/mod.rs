pub mod defaults;
pub mod traits;

use std::{env, pin::Pin, sync::Arc};

use crate::{
    conversion::request::Request,
    models::{openai::OpenAIMessage, openai::OpenAIStreamChunk},
};
use futures_util::Stream;

use self::{
    defaults::{
        DefaultSystemPromptAdapter, DefaultToolsAdapter, DefaultUserPromptAdapter,
        GeminiToolSchemaAdapter, KimiMaxTokensAdapter, OAIReasoningModelAdapter,
        ToolSimulationModelAdapter, ToolSimulationRequestAdapter, ToolSimulationResponseAdapter,
        ToolSimulationToolAdapter,
    },
    traits::Adapter,
};

pub struct RequestAdapter {
    adapters: Vec<Arc<dyn Adapter>>,
}

impl RequestAdapter {
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let mut adapters: Vec<Arc<dyn Adapter>> = Vec::new();

        adapters.push(Arc::new(DefaultSystemPromptAdapter));
        adapters.push(Arc::new(DefaultUserPromptAdapter));
        adapters.push(Arc::new(DefaultToolsAdapter));

        let is_gemini = model.contains("gemini");
        let is_kimi = model.contains("moonshotai/kimi-k2-instruct")
            && !env::var("DISABLE_GROQ_MAX_TOKENS")
                .is_ok_and(|value| value == "1" || value == "true");

        if is_gemini {
            adapters.push(Arc::new(GeminiToolSchemaAdapter));
        }
        if is_kimi {
            adapters.push(Arc::new(KimiMaxTokensAdapter));
        }

        adapters.push(Arc::new(OAIReasoningModelAdapter));

        if model.ends_with("-xml-tools") || model.ends_with("-bracket-tools") {
            adapters.push(Arc::new(ToolSimulationRequestAdapter));
            adapters.push(Arc::new(ToolSimulationResponseAdapter));
            adapters.push(Arc::new(ToolSimulationModelAdapter));
            adapters.push(Arc::new(ToolSimulationToolAdapter));
        }

        Self { adapters }
    }

    #[must_use]
    pub fn adapt_user_prompt(&self, user_prompt: &str, request: &Request) -> String {
        self.adapters
            .iter()
            .fold(user_prompt.to_string(), |prompt, adapter| {
                adapter.adapt_user_prompt(&prompt, request)
            })
    }

    #[must_use]
    pub fn adapt_system_prompt(&self, system_prompt: &str, request: &Request) -> String {
        self.adapters
            .iter()
            .fold(system_prompt.to_string(), |prompt, adapter| {
                adapter.adapt_system_prompt(&prompt, request)
            })
    }

    #[must_use]
    pub fn adapt_messages(
        &self,
        messages: Vec<OpenAIMessage>,
        request: &Request,
    ) -> Vec<OpenAIMessage> {
        self.adapters
            .iter()
            .fold(messages, |messages, adapter| {
                adapter.adapt_messages(messages, request)
            })
    }

    #[must_use]
    pub fn adapt_model(&self, model: &str, request: &Request) -> String {
        self.adapters
            .iter()
            .fold(model.to_string(), |model, adapter| {
                adapter.adapt_model(&model, request)
            })
    }

    #[must_use]
    pub fn adapt_tools(
        &self,
        tools: Option<Vec<crate::models::claude::ClaudeTool>>,
        request: &Request,
    ) -> Option<Vec<crate::models::claude::ClaudeTool>> {
        self.adapters
            .iter()
            .fold(tools, |tools, adapter| adapter.adapt_tools(tools, request))
    }

    #[must_use]
    pub fn adapt_tool_choice(
        &self,
        tool_choice: Option<crate::models::claude::ClaudeToolChoice>,
        request: &Request,
    ) -> Option<crate::models::claude::ClaudeToolChoice> {
        self.adapters
            .iter()
            .fold(tool_choice, |tool_choice, adapter| {
                adapter.adapt_tool_choice(tool_choice, request)
            })
    }

    #[must_use]
    pub fn adapt_temperature(&self, temperature: Option<f32>, request: &Request) -> Option<f32> {
        self.adapters
            .iter()
            .fold(temperature, |temperature, adapter| {
                adapter.adapt_temperature(temperature, request)
            })
    }

    #[must_use]
    pub fn adapt_top_p(&self, top_p: Option<f32>, request: &Request) -> Option<f32> {
        self.adapters
            .iter()
            .fold(top_p, |top_p, adapter| adapter.adapt_top_p(top_p, request))
    }

    #[must_use]
    pub fn adapt_max_tokens(&self, max_tokens: u32, request: &Request) -> Option<u32> {
        self.adapters
            .iter()
            .try_fold(max_tokens, |max_tokens, adapter| {
                adapter.adapt_max_tokens(max_tokens, request)
            })
    }

    #[must_use]
    pub fn adapt_max_completion_tokens(&self, max_tokens: u32, request: &Request) -> Option<u32> {
        self.adapters.iter().fold(None, |accumulator, adapter| {
            accumulator.or_else(|| adapter.adapt_max_completion_tokens(max_tokens, request))
        })
    }

    #[must_use]
    pub fn adapt_tool_result(
        &self,
        tool_name: &str,
        tool_result: &str,
        request: &Request,
    ) -> String {
        self.adapters
            .iter()
            .fold(tool_result.to_string(), |tool_result, adapter| {
                adapter.adapt_tool_result(tool_name, &tool_result, request)
            })
    }

    #[must_use]
    pub fn adapt_tool_schema(
        &self,
        schema: &serde_json::Value,
        request: &Request,
    ) -> serde_json::Value {
        self.adapters
            .iter()
            .fold(schema.clone(), |schema, adapter| {
                adapter.adapt_tool_schema(&schema, request)
            })
    }

    #[must_use]
    pub fn adapt_tool_description(&self, description: &str, request: &Request) -> String {
        self.adapters
            .iter()
            .fold(description.to_string(), |description, adapter| {
                adapter.adapt_tool_description(&description, request)
            })
    }

    #[must_use]
    pub fn adapt_non_stream_response(
        &self,
        response: serde_json::Value,
        request: &Request,
    ) -> serde_json::Value {
        self.adapters.iter().fold(response, |response, adapter| {
            adapter.adapt_non_stream_response(response, request)
        })
    }

    #[must_use]
    pub fn adapt_chunk_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
        self.adapters.iter().fold(stream, |stream, adapter| {
            adapter.adapt_chunk_stream(stream, req)
        })
    }
}
