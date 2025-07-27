use ant_compat::{
    adapters::{defaults::DefaultToolsAdapter, traits::Adapter},
    conversion::request::Request,
    models::claude::ClaudeMessagesRequest,
};
use insta::assert_snapshot;
use rstest::rstest;
use saphyr::{LoadableYamlNode, ScalarOwned, YamlOwned};
use std::{fs, path::Path};

fn load_descriptions() -> Option<YamlOwned> {
    let path = Path::new("tests/fixtures/tool_descriptions.yaml");
    if let Ok(content) = fs::read_to_string(path) {
        if let Ok(mut docs) = YamlOwned::load_from_str(&content) {
            return Some(docs.remove(0));
        }
    }
    None
}

fn dummy_request() -> Request {
    let req = ClaudeMessagesRequest {
        model: "qwen3-coder-plus".to_string(),
        messages: vec![],
        system: None,
        max_tokens: 65536,
        stop_sequences: None,
        stream: None,
        temperature: Some(0.7),
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };
    req.into()
}

struct TestGrepAdapter;
impl Adapter for TestGrepAdapter {
    fn adapt_tool_description(&self, description: &str, _request: &Request) -> String {
        description.replace("Grep", "RipGrep")
    }
}

#[rstest]
fn test_grep_tool_description() {
    if let Some(YamlOwned::Mapping(descs)) = load_descriptions() {
        let key = YamlOwned::Value(ScalarOwned::String("grep".to_string()));
        if let Some(desc) = descs.get(&key).and_then(|v| v.as_str()) {
            let adapter = TestGrepAdapter;
            let adapted = adapter.adapt_tool_description(desc, &dummy_request());
            assert_snapshot!("grep_tool_description", adapted);
        }
    }
}

#[rstest]
#[case("multiedit")]
#[case("write")]
#[case("edit")]
fn test_default_tool_descriptions(#[case] tool_name: &str) {
    if let Some(YamlOwned::Mapping(descs)) = load_descriptions() {
        let key = YamlOwned::Value(ScalarOwned::String(tool_name.to_string()));
        if let Some(description) = descs.get(&key).and_then(|v| v.as_str()) {
            let adapter = DefaultToolsAdapter;
            let adapted = adapter.adapt_tool_description(description, &dummy_request());
            assert_snapshot!(format!("{tool_name}_description"), adapted);
        }
    }
}
