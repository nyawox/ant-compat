pub mod defaults;
pub mod traits;

use std::{env, sync::Arc};

use self::{
    defaults::{
        DefaultSystemPromptAdapter, DefaultToolsAdapter, DefaultUserPromptAdapter,
        GeminiToolSchemaAdapter, KimiMaxTokensAdapter,
    },
    traits::Adapter,
};

pub struct RequestAdapter(pub Arc<dyn Adapter>);

impl RequestAdapter {
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let base_adapter = (
            DefaultToolsAdapter,
            (DefaultSystemPromptAdapter, DefaultUserPromptAdapter),
        );

        if model.contains("gemini") {
            return Self(Arc::new((GeminiToolSchemaAdapter, base_adapter)));
        }
        if model.contains("moonshotai/kimi-k2-instruct")
            && !env::var("DISABLE_GROQ_MAX_TOKENS").is_ok_and(|v| v == "1" || v == "true")
        {
            return Self(Arc::new((KimiMaxTokensAdapter, base_adapter)));
        }

        Self(Arc::new(base_adapter))
    }
}
