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
        GeminiToolSchemaAdapter, KimiMaxTokensAdapter, PseudoFunctionAdapter,
        PseudoFunctionModelAdapter, PseudoFunctionResponseAdapter, PseudoFunctionToolAdapter,
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
            && !env::var("DISABLE_GROQ_MAX_TOKENS").is_ok_and(|v| v == "1" || v == "true");

        if is_gemini {
            adapters.push(Arc::new(GeminiToolSchemaAdapter));
        }
        if is_kimi {
            adapters.push(Arc::new(KimiMaxTokensAdapter));
        }

        if model.ends_with("-xml-tools") || model.ends_with("-bracket-tools") {
            adapters.push(Arc::new(PseudoFunctionAdapter));
            adapters.push(Arc::new(PseudoFunctionModelAdapter));
            adapters.push(Arc::new(PseudoFunctionToolAdapter));
            adapters.push(Arc::new(PseudoFunctionResponseAdapter));
        }

        Self { adapters }
    }

    #[must_use]
    pub fn adapt_user_prompt(&self, user_prompt: &str, req: &Request) -> String {
        self.adapters
            .iter()
            .fold(user_prompt.to_string(), |prompt, adapter| {
                adapter.adapt_user_prompt(&prompt, req)
            })
    }

    #[must_use]
    pub fn adapt_system_prompt(&self, system_prompt: &str, req: &Request) -> String {
        self.adapters
            .iter()
            .fold(system_prompt.to_string(), |prompt, adapter| {
                adapter.adapt_system_prompt(&prompt, req)
            })
    }

    #[must_use]
    pub fn adapt_messages(
        &self,
        messages: Vec<OpenAIMessage>,
        req: &Request,
    ) -> Vec<OpenAIMessage> {
        self.adapters
            .iter()
            .fold(messages, |msgs, adapter| adapter.adapt_messages(msgs, req))
    }

    #[must_use]
    pub fn adapt_model(&self, model: &str, req: &Request) -> String {
        self.adapters
            .iter()
            .fold(model.to_string(), |m, adapter| adapter.adapt_model(&m, req))
    }

    #[must_use]
    pub fn adapt_tools(
        &self,
        tools: Option<Vec<crate::models::claude::ClaudeTool>>,
        req: &Request,
    ) -> Option<Vec<crate::models::claude::ClaudeTool>> {
        self.adapters
            .iter()
            .fold(tools, |t, adapter| adapter.adapt_tools(t, req))
    }

    #[must_use]
    pub fn adapt_tool_choice(
        &self,
        tool_choice: Option<crate::models::claude::ClaudeToolChoice>,
        req: &Request,
    ) -> Option<crate::models::claude::ClaudeToolChoice> {
        self.adapters.iter().fold(tool_choice, |tc, adapter| {
            adapter.adapt_tool_choice(tc, req)
        })
    }

    #[must_use]
    pub fn adapt_temperature(&self, temperature: Option<f32>, req: &Request) -> Option<f32> {
        self.adapters.iter().fold(temperature, |temp, adapter| {
            adapter.adapt_temperature(temp, req)
        })
    }

    #[must_use]
    pub fn adapt_top_p(&self, top_p: Option<f32>, req: &Request) -> Option<f32> {
        self.adapters
            .iter()
            .fold(top_p, |p, adapter| adapter.adapt_top_p(p, req))
    }

    #[must_use]
    pub fn adapt_max_tokens(&self, max_tokens: u32, req: &Request) -> u32 {
        self.adapters
            .iter()
            .fold(max_tokens, |m, adapter| adapter.adapt_max_tokens(m, req))
    }

    #[must_use]
    pub fn adapt_tool_result(&self, tool_name: &str, tool_result: &str, req: &Request) -> String {
        self.adapters
            .iter()
            .fold(tool_result.to_string(), |result, adapter| {
                adapter.adapt_tool_result(tool_name, &result, req)
            })
    }

    #[must_use]
    pub fn adapt_tool_schema(
        &self,
        schema: &serde_json::Value,
        req: &Request,
    ) -> serde_json::Value {
        self.adapters
            .iter()
            .fold(schema.clone(), |schema_val, adapter| {
                adapter.adapt_tool_schema(&schema_val, req)
            })
    }

    #[must_use]
    pub fn adapt_tool_description(&self, description: &str, req: &Request) -> String {
        self.adapters
            .iter()
            .fold(description.to_string(), |desc, adapter| {
                adapter.adapt_tool_description(&desc, req)
            })
    }

    #[must_use]
    pub fn adapt_non_stream_response(
        &self,
        response: serde_json::Value,
        req: &Request,
    ) -> serde_json::Value {
        self.adapters.iter().fold(response, |resp, adapter| {
            adapter.adapt_non_stream_response(resp, req)
        })
    }

    #[must_use]
    pub fn adapt_chunk_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
        self.adapters.iter().fold(stream, |strm, adapter| {
            adapter.adapt_chunk_stream(strm, req)
        })
    }
}
