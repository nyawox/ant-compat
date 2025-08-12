use crate::{
    adapters::traits::Adapter,
    conversion::request::Request,
    models::claude::{ClaudeTool, ClaudeToolChoice},
};

pub struct ToolSimulationToolAdapter;

impl Adapter for ToolSimulationToolAdapter {
    fn adapt_tools(
        &self,
        _tools: Option<Vec<ClaudeTool>>,
        _request: &Request,
    ) -> Option<Vec<ClaudeTool>> {
        None
    }

    fn adapt_tool_choice(
        &self,
        _tool_choice: Option<ClaudeToolChoice>,
        _request: &Request,
    ) -> Option<ClaudeToolChoice> {
        None
    }
}
