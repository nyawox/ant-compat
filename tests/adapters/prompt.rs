use ant_compat::{
    adapters::{defaults::DefaultSystemPromptAdapter, traits::Adapter},
    conversion::request::Request,
    models::claude::ClaudeMessagesRequest,
};
use insta::assert_snapshot;
use regex::Regex;
use rstest::rstest;
use saphyr::{LoadableYamlNode, ScalarOwned, YamlOwned};
use std::{fs, path::Path};

fn load_prompt() -> Option<YamlOwned> {
    let path = Path::new("tests/fixtures/system_prompt.yaml");
    if let Ok(content) = fs::read_to_string(path) {
        if let Ok(mut docs) = YamlOwned::load_from_str(&content) {
            return Some(docs.remove(0));
        }
    }
    None
}

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

struct RedactionAdapter;
impl Adapter for RedactionAdapter {
    fn adapt_user_prompt(&self, prompt: &str, _request: &Request) -> String {
        if let Ok(re) = Regex::new(r"AIzaSy[\w\-]{33,34}") {
            re.replace_all(prompt, "[REDACTED]").to_string()
        } else {
            prompt.to_string()
        }
    }
}

#[rstest]
fn test_system_prompt() {
    if let Some(YamlOwned::Mapping(map)) = load_prompt() {
        if let Some(YamlOwned::Value(ScalarOwned::String(prompt))) =
            map.get(&YamlOwned::Value(ScalarOwned::String("prompt".into())))
        {
            let adapter = DefaultSystemPromptAdapter;
            let request = dummy_request();
            let result = adapter.adapt_system_prompt(prompt, &request);
            assert_snapshot!(result);
        }
    }
}

#[rstest]
fn test_system_prompt_oai() {
    if let Some(YamlOwned::Mapping(map)) = load_prompt() {
        if let Some(YamlOwned::Value(ScalarOwned::String(prompt))) =
            map.get(&YamlOwned::Value(ScalarOwned::String("prompt".into())))
        {
            let adapter = DefaultSystemPromptAdapter;
            let request = dummy_oai_request();
            let result = adapter.adapt_system_prompt(prompt, &request);
            assert_snapshot!(result);
        }
    }
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
