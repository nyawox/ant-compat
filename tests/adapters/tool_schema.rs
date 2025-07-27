use ant_compat::{
    adapters::{defaults::GeminiToolSchemaAdapter, traits::Adapter},
    models::claude::ClaudeMessagesRequest,
};
use insta::assert_snapshot;
use rstest::rstest;
use serde_json::{json, to_string_pretty};

fn dummy_request() -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: "llama-4-mediocre-17B-128E-Instruct".to_string(),
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

#[rstest]
#[case(
    "gemini_unsupported_schema_1",
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "format": "email"
            },
            "date": {
                "type": "string",
                "format": "date-time"
            }
        },
        "additionalProperties": false
    })
)]
#[case(
    "gemini_unsupported_schema_2",
    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "User",
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"}
                }
            }
        },
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"]
            },
            "details": {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer"}
                        }
                    },
                    {
                        "$ref": "#/definitions/Address"
                    }
                ]
            }
        },
        "additionalProperties": false
    })
)]
fn test_gemini_schema_cleaning(#[case] name: &str, #[case] schema: serde_json::Value) {
    let adapter = GeminiToolSchemaAdapter;
    let request = dummy_request();
    let cleaned = adapter.adapt_tool_schema(&schema, &request);
    let snapshot =
        to_string_pretty(&cleaned).expect("Serialization of a JSON value should not fail");
    assert_snapshot!(name, snapshot);
}
