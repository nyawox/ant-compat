use crate::helpers::load_system_prompt_fixture;
use ant_compat::{
    adapters::{defaults::DefaultSystemPromptAdapter, traits::Adapter},
    conversion::request::Request,
    lazy_regex,
    models::claude::ClaudeMessagesRequest,
};
use insta::assert_snapshot;
use regex::Regex;
use rstest::rstest;
use std::sync::LazyLock;

fn dummy_request() -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: "google/gemini-3.0-pro-beta".to_string(),
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

fn dummy_oai_request() -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: "openai/o3".to_string(),
        messages: vec![],
        system: None,
        max_tokens: 65536,
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

struct UppercaseAdapter;
impl Adapter for UppercaseAdapter {
    fn adapt_system_prompt(&self, prompt: &str, _request: &Request) -> String {
        prompt.to_uppercase()
    }
}

static REDACTION_REGEX: LazyLock<Regex> = lazy_regex!(r"AIzaSy[\w\-]{33,34}");

struct RedactionAdapter;
impl Adapter for RedactionAdapter {
    fn adapt_user_prompt(&self, prompt: &str, _request: &Request) -> String {
        REDACTION_REGEX
            .replace_all(prompt, "[REDACTED]")
            .to_string()
    }
}

#[rstest]
fn test_system_prompt() {
    let prompt = load_system_prompt_fixture();
    let adapter = DefaultSystemPromptAdapter;
    let request = dummy_request();
    let result = adapter.adapt_system_prompt(&prompt, &request);
    assert_snapshot!(result);
}

#[rstest]
fn test_system_prompt_oai() {
    let prompt = load_system_prompt_fixture();
    let adapter = DefaultSystemPromptAdapter;
    let request = dummy_oai_request();
    let result = adapter.adapt_system_prompt(&prompt, &request);
    assert_snapshot!(result);
}

#[test]
fn test_uppercase_adapter() {
    let adapter = UppercaseAdapter;
    let request = dummy_request();
    let result = adapter.adapt_system_prompt("hello world", &request);
    assert_snapshot!(result);
}

#[test]
fn test_redaction_adapter() {
    let adapter = RedactionAdapter;
    let request = dummy_request();
    let result = adapter.adapt_user_prompt(
        "GEMINI_API_KEY=AIzaSyToTaLLyNoTAReAlKey12345678AizaSk-",
        &request,
    );
    assert_snapshot!(result);
}
