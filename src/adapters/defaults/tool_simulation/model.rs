use crate::{adapters::traits::Adapter, conversion::request::Request};

pub struct ToolSimulationModelAdapter;

impl Adapter for ToolSimulationModelAdapter {
    fn adapt_model(&self, model: &str, _request: &Request) -> String {
        model
            .strip_suffix("-xml-tools")
            .or_else(|| model.strip_suffix("-bracket-tools"))
            .unwrap_or(model)
            .to_string()
    }
}
