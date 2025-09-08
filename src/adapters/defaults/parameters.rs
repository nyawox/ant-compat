use crate::{adapters::traits::Adapter, conversion::request::Request};

pub struct KimiMaxTokensAdapter;

// workaround for groq
impl Adapter for KimiMaxTokensAdapter {
    fn adapt_max_tokens(&self, _max_tokens: u32, _request: &Request) -> Option<u32> {
        Some(16384)
    }
}

fn is_openai_reasoning_model(model: &str) -> bool {
    matches!(model, "o3" | "o3-mini" | "o4-mini")
        || model.contains("gpt-5")
        || model.contains("openai")
}

pub struct OAIReasoningModelAdapter;

impl Adapter for OAIReasoningModelAdapter {
    fn adapt_max_tokens(&self, max_tokens: u32, request: &Request) -> Option<u32> {
        if is_openai_reasoning_model(&request.model) {
            None
        } else {
            Some(max_tokens)
        }
    }

    fn adapt_max_completion_tokens(
        &self,
        claude_max_tokens: u32,
        request: &Request,
    ) -> Option<u32> {
        if is_openai_reasoning_model(&request.model) {
            Some(claude_max_tokens)
        } else {
            None
        }
    }
}
