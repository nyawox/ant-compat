use serde_json::Value;

use crate::conversion::request::Request;

pub trait Adapter: Send + Sync {
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
}

impl<A: Adapter, B: Adapter> Adapter for (A, B) {
    fn adapt_tool_schema(&self, schema: &Value, req: &Request) -> Value {
        self.1
            .adapt_tool_schema(&self.0.adapt_tool_schema(schema, req), req)
    }
    fn adapt_tool_description(&self, d: &str, req: &Request) -> String {
        self.1
            .adapt_tool_description(&self.0.adapt_tool_description(d, req), req)
    }
    fn adapt_system_prompt(&self, p: &str, req: &Request) -> String {
        self.1
            .adapt_system_prompt(&self.0.adapt_system_prompt(p, req), req)
    }
    fn adapt_user_prompt(&self, p: &str, req: &Request) -> String {
        self.1
            .adapt_user_prompt(&self.0.adapt_user_prompt(p, req), req)
    }
    fn adapt_temperature(&self, t: Option<f32>, req: &Request) -> Option<f32> {
        self.1
            .adapt_temperature(self.0.adapt_temperature(t, req), req)
    }
    fn adapt_top_p(&self, p: Option<f32>, req: &Request) -> Option<f32> {
        self.1.adapt_top_p(self.0.adapt_top_p(p, req), req)
    }
    fn adapt_max_tokens(&self, m: u32, req: &Request) -> u32 {
        self.1
            .adapt_max_tokens(self.0.adapt_max_tokens(m, req), req)
    }
    fn adapt_tool_result(&self, tool_name: &str, tool_result: &str, req: &Request) -> String {
        self.1.adapt_tool_result(
            tool_name,
            &self.0.adapt_tool_result(tool_name, tool_result, req),
            req,
        )
    }
}
