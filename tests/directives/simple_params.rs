use crate::helpers::load_system_prompt_fixture;
use ant_compat::{
    directives::processor::DirectiveProcessor,
    models::claude::{ClaudeMessagesRequest, ClaudeSystem},
};
use insta::assert_debug_snapshot;
use rstest::rstest;

fn base_request() -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: "hype-ultraman".to_string(),
        messages: vec![],
        system: None,
        max_tokens: 4068,
        temperature: Some(1.0),
        top_p: Some(0.7),
        stream: None,
        stop_sequences: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    }
}

#[rstest]
#[case(
    "simple",
    r#"--- PROXY DIRECTIVE ---
{
    "rules": [
        {
            "if": { "modelContains": "hype-ultraman" },
            "apply": {
                "model": "proxy-directive-test-passed",
                "max_tokens": 65536,
                "temperature": 0.7,
                "top_p": 0.8
            }
        }
    ]
}
--- END DIRECTIVE ---"#
)]
#[case("no_changes", "")]
#[case(
    "global_setting",
    r#"--- PROXY DIRECTIVE ---
{
    "global": {
        "model": "proxy-global-directive"
    }
}
--- END DIRECTIVE ---"#
)]
#[case(
    "non_matching_rule",
    r#"--- PROXY DIRECTIVE ---
{
    "rules": [
        {
            "if": { "modelContains": "some-other-model" },
            "apply": {
                "model": "this-should-not-be-applied"
            }
        }
    ]
}
--- END DIRECTIVE ---"#
)]
#[case(
    "global_plus_matching_rule",
    r#"--- PROXY DIRECTIVE ---
{
    "global": {
        "temperature": 0.5
    },
    "rules": [
        {
            "if": { "modelContains": "hype-ultraman" },
            "apply": {
                "model": "rule-overrides-model"
            }
        }
    ]
}
--- END DIRECTIVE ---"#
)]
fn test_proxy_directive_processing(#[case] name: &str, #[case] directive_block: &str) {
    let mut request = base_request();
    let base_prompt = load_system_prompt_fixture();
    let system_prompt = if directive_block.is_empty() {
        base_prompt
    } else {
        format!("{directive_block}\n{base_prompt}")
    };
    request.system = Some(ClaudeSystem::Text(system_prompt));

    DirectiveProcessor::process(&mut request);

    assert_debug_snapshot!(name, request);
}
