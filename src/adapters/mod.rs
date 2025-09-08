pub mod defaults;
pub mod traits;

use std::{env, pin::Pin, sync::Arc};

use crate::{
    conversion::request::Request,
    conversion::stream::{chunks_to_events, emit_event, emit_ping},
    directives::models::Settings,
    error::AppError,
    models::{
        claude::ClaudeMessagesRequest, openai::OpenAIMessage, openai::OpenAIRequest,
        openai::OpenAIStreamChunk,
    },
    state::AppState,
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use self::{
    defaults::{
        DefaultSystemPromptAdapter, DefaultToolsAdapter, DefaultUserPromptAdapter,
        GeminiToolSchemaAdapter, KimiMaxTokensAdapter, OAIReasoningModelAdapter,
        ResponsesApiAdapter, ThreadOfMeowsingsAdapter, ToolSimulationModelAdapter,
        ToolSimulationRequestAdapter, ToolSimulationResponseAdapter, ToolSimulationToolAdapter,
    },
    traits::{Adapter, ApiAdapter},
};

pub struct RequestAdapter {
    adapters: Vec<Arc<dyn Adapter>>,
    api: Option<Arc<dyn ApiAdapter>>,
}

impl RequestAdapter {
    #[must_use]
    pub fn for_model(model: &str, settings: &Settings) -> Self {
        let mut adapters: Vec<Arc<dyn Adapter>> = Vec::new();

        let disable_defaults =
            env::var("DISABLE_DEFAULT_ADAPTERS").is_ok_and(|value| value == "1" || value == "true");

        if !disable_defaults {
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
        }

        if settings.enable_meowsings.unwrap_or(false) {
            adapters.push(Arc::new(ThreadOfMeowsingsAdapter));
        }

        if model.ends_with("-xml-tools") || model.ends_with("-bracket-tools") {
            adapters.push(Arc::new(ToolSimulationRequestAdapter));
            adapters.push(Arc::new(ToolSimulationResponseAdapter));
            adapters.push(Arc::new(ToolSimulationModelAdapter));
            adapters.push(Arc::new(ToolSimulationToolAdapter));
        }

        let api = match settings.responses.as_ref() {
            Some(responses_settings) if responses_settings.enable.unwrap_or(false) => {
                Some(Arc::new(ResponsesApiAdapter {
                    max_output_tokens: responses_settings.max_output_tokens,
                    reasoning_summary: responses_settings.reasoning_summary.clone(),
                }) as Arc<dyn ApiAdapter>)
            }
            _ => None,
        };

        Self { adapters, api }
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
    pub fn endpoint_suffix(&self) -> &'static str {
        self.api
            .as_ref()
            .map_or("/chat/completions", |api| api.endpoint_suffix())
    }

    #[must_use]
    pub fn build_request_body(&self, openai_req: &OpenAIRequest, original: &Request) -> Value {
        match self.api.as_ref() {
            Some(api) => api.build_body(openai_req, original),
            None => serde_json::to_value(openai_req).unwrap_or_else(|_| json!({})),
        }
    }

    #[must_use]
    pub fn normalize_non_stream_json(&self, response_json: Value, _original: &Request) -> Value {
        match self.api.as_ref() {
            Some(api) => api.normalize_non_stream_json(response_json),
            None => response_json,
        }
    }

    #[must_use]
    pub fn build_anthropic_sse_stream(
        &self,
        response: reqwest::Response,
        target_model: &str,
        original: &ClaudeMessagesRequest,
        state: &AppState,
    ) -> Pin<Box<dyn Stream<Item = Result<Bytes, AppError>> + Send>> {
        match self.api.as_ref() {
            Some(api) => {
                let chunk_stream = api.chunk_stream(response, original);
                let adapted_chunks = self.adapt_chunk_stream(Box::pin(chunk_stream), original);
                let event_stream =
                    chunks_to_events(target_model, adapted_chunks, state.idle_connection_timeout);
                Box::pin(async_stream::stream! {
                    let mut stream = Box::pin(event_stream);
                    let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));
                    loop {
                        tokio::select! {
                            event_result = stream.next() => {
                                if let Some(event_result) = event_result {
                                    match event_result {
                                        Ok(event) => {
                                            let (event_type, data) = event.to_parts();
                                            yield Ok(emit_event(event_type, &data));
                                        }
                                        Err(e) => {
                                            let error_event = json!({
                                                "type": "error",
                                                "error": { "type": "api_error", "message": e.to_string() }
                                            });
                                            yield Ok(emit_event("error", &error_event));
                                            break;
                                        }
                                    }
                                } else {
                                    break;
                                }
                            }
                            _ = ping_interval.tick() => {
                                yield Ok(emit_ping());
                            }
                        }
                    }
                })
            }
            None => crate::conversion::convert_openai_stream_to_anthropic(
                response,
                target_model,
                self,
                original,
                state,
            ),
        }
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
        self.adapters.iter().fold(messages, |messages, adapter| {
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
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>>,
        request: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>> {
        self.adapters.iter().fold(stream, |stream, adapter| {
            adapter.adapt_chunk_stream(stream, request)
        })
    }
}
