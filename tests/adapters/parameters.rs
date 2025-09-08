use ant_compat::{
    adapters::traits::Adapter, conversion::request::Request, models::claude::ClaudeMessagesRequest,
};
use insta::assert_debug_snapshot;
use rstest::rstest;

fn dummy_request() -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: "test-model".to_string(),
        messages: vec![],
        system: None,
        max_tokens: 1024,
        stop_sequences: None,
        stream: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    }
}

struct FixedTempAdapter(f32);
impl Adapter for FixedTempAdapter {
    fn adapt_temperature(&self, _temperature: Option<f32>, _request: &Request) -> Option<f32> {
        Some(self.0)
    }
}

struct ClampTopPAdapter {
    min: f32,
    max: f32,
}
impl Adapter for ClampTopPAdapter {
    fn adapt_top_p(&self, top_p: Option<f32>, _request: &Request) -> Option<f32> {
        top_p.map(|p| p.clamp(self.min, self.max))
    }
}

struct AddTokensAdapter(u32);
impl Adapter for AddTokensAdapter {
    fn adapt_max_tokens(&self, max_tokens: u32, _request: &Request) -> Option<u32> {
        Some(max_tokens + self.0)
    }
}

#[rstest]
fn test_temp_adapter() {
    let adapter = FixedTempAdapter(0.99);
    let request = dummy_request();
    assert_debug_snapshot!(adapter.adapt_temperature(Some(0.5), &request));
}

#[rstest]
#[case::low(Some(0.0), "low")]
#[case::mid(Some(0.7), "mid")]
#[case::high(Some(1.2), "high")]
#[case::none(None, "none")]
fn test_top_p_adapter(#[case] input: Option<f32>, #[case] name: &str) {
    let adapter = ClampTopPAdapter { min: 0.1, max: 0.9 };
    let request = dummy_request();
    assert_debug_snapshot!(name, adapter.adapt_top_p(input, &request));
}

#[rstest]
fn test_max_tokens_adapter() {
    let adapter = AddTokensAdapter(100);
    let request = dummy_request();
    assert_debug_snapshot!(adapter.adapt_max_tokens(1024, &request));
}
