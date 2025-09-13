use crate::helpers;
use ant_compat::adapters::defaults::tool_simulation::parsing::parse_bracket_tool;
use ant_compat::{
    adapters::RequestAdapter,
    conversion::{request::convert_claude_to_openai, stream::convert_openai_stream_to_anthropic},
    directives::models::Settings,
    models::{
        claude::{
            AnthropicStreamEvent, ClaudeContent, ClaudeContentBlock, ClaudeMessage,
            ClaudeMessagesRequest, ClaudeSystem, ClaudeTool, ClaudeToolChoice, ContentBlock,
        },
        openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIUsage},
    },
};
use rstest::rstest;
use serde_json::{Value, json};

fn redact_ids(events: &mut [AnthropicStreamEvent]) {
    for event in events.iter_mut() {
        match event {
            AnthropicStreamEvent::MessageStart(start) => {
                start.message.id = "[redacted-id]".to_string();
            }
            AnthropicStreamEvent::ContentBlockStart(start) => {
                if let ContentBlock::ToolUse { id, .. } = &mut start.content_block {
                    *id = "[redacted-tool-id]".to_string();
                }
            }
            _ => {}
        }
    }
}

fn text_content_block(text: &str) -> ClaudeContentBlock {
    ClaudeContentBlock {
        block_type: "text".to_string(),
        text: Some(text.to_string()),
        source: None,
        id: None,
        name: None,
        input: None,
        tool_use_id: None,
        content: None,
    }
}

fn tool_use_content_block(id: &str, name: &str, input: Value) -> ClaudeContentBlock {
    ClaudeContentBlock {
        block_type: "tool_use".to_string(),
        text: None,
        source: None,
        id: Some(id.to_string()),
        name: Some(name.to_string()),
        input: Some(input),
        tool_use_id: None,
        content: None,
    }
}

fn tool_result_content_block(tool_use_id: &str, content: Value) -> ClaudeContentBlock {
    ClaudeContentBlock {
        block_type: "tool_result".to_string(),
        text: None,
        source: None,
        id: None,
        name: None,
        input: None,
        tool_use_id: Some(tool_use_id.to_string()),
        content: Some(content),
    }
}

fn get_request(model: &str) -> ClaudeMessagesRequest {
    ClaudeMessagesRequest {
        model: model.to_string(),
        messages: vec![
            ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Text(
                    "What is the weather like in San Francisco?".to_string(),
                ),
            },
            ClaudeMessage {
                role: "assistant".to_string(),
                content: ClaudeContent::Array(vec![
                    text_content_block("I'll get the weather for you."),
                    tool_use_content_block(
                        "toolu_123",
                        "get_weather",
                        json!({"location": "San Francisco"}),
                    ),
                ]),
            },
            ClaudeMessage {
                role: "user".to_string(),
                content: ClaudeContent::Array(vec![tool_result_content_block(
                    "toolu_123",
                    json!({"temperature": "72F"}),
                )]),
            },
        ],
        system: Some(ClaudeSystem::Text(
            "You are a helpful assistant.".to_string(),
        )),
        max_tokens: 65536,
        stop_sequences: None,
        stream: Some(true),
        temperature: Some(0.5),
        top_p: Some(1.0),
        top_k: None,
        tools: Some(vec![ClaudeTool {
            name: "get_weather".to_string(),
            description: Some("Get the current weather in a given location.".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        }]),
        tool_choice: Some(ClaudeToolChoice {
            choice_type: "auto".to_string(),
            name: None,
        }),
        thinking: None,
    }
}

#[rstest]
#[case("request_conversion", "google/gemini-2.5-pro-xml-tools")]
#[case("bracket_tools_conversion", "google/gemini-2.5-pro-bracket-tools")]
fn verify_request_conversion(#[case] snapshot_name: &str, #[case] model: &str) {
    let request = get_request(model);
    let adapter = RequestAdapter::for_model(model, &Settings::default());
    let result = convert_claude_to_openai(request, model, &adapter);
    insta::assert_debug_snapshot!(snapshot_name, &result);
}

async fn run_stream_conversion_test(
    model: &str,
    chunks: Vec<OpenAIStreamChunk>,
    snapshot_name: &str,
) {
    let request = get_request(model);
    let mock_response = helpers::mock_response_from_chunks(chunks).await;
    let adapter = RequestAdapter::for_model(model, &Settings::default());
    let mock_state = helpers::mock_app_state();

    let anthropic_stream =
        convert_openai_stream_to_anthropic(mock_response, model, &adapter, &request, &mock_state);

    let mut events = helpers::collect_and_parse_stream(anthropic_stream).await;
    redact_ids(&mut events);

    insta::assert_debug_snapshot!(snapshot_name, events);
}

#[tokio::test]
async fn verify_xml_tools_stream_conversion() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("Thinking about the weather... ".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("<function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"<invoke name="get_weather">"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#"<parameter name="location">San Francisco</parameter>"#.to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "5".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("</invoke></function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "6".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta::default(),
                finish_reason: Some("stop".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-xml-tools",
        chunks,
        "xml_tools_stream_conversion",
    )
    .await;
}

#[tokio::test]
async fn verify_xml_tools_json_repair() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("<function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"<invoke name="MultiEdit">"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"<parameter name="edits">"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#"```json
[{"old_string":"a","new_string":"b",}]
```"#
                            .to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "5".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("</parameter></invoke></function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-xml-tools",
        chunks,
        "xml_tools_json_repair",
    )
    .await;
}

#[tokio::test]
async fn verify_xml_tools_fragmented_delimiter() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("Thinking... <function".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        "_calls><invoke name=\"get_weather\"><parameter name=\"location\">SF</parameter></invoke></function_calls>".to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-xml-tools",
        chunks,
        "xml_tools_fragmented_delimiter",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_stream_conversion() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("Thinking...\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---TOOLS---\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("[tool(get_weather, location=\"San Francisco\")]\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        "[tool(get_weather, location=\"\"\"Germany/FrankFurt\"\"\")]\n".to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "5".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("[tool(get_weather,".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "6".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("location=\"".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "7".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("\"\"Norway/Oslo\"\"\")]\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "8".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---END_TOOLS---".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_stream_conversion",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_complex_parsing() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---TOOLS---n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#"[tool(Test, json="""{"key": "value with quotes"}""""#.to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#", path="""/var/lib/secrets/postgres-production.yaml""", edits="""[{old_string="failing_test", new_string="fix_by_removing_the_test"}]""")]"#.to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"[tool(NoArgs)]"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "5".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"[tool(WithParens, arg="(value)")]"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "6".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---END_TOOLS---".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_complex_parsing",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_fragmented_delimiter() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("Thinking... ---TO".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        "OLS---[tool(get_weather, location=SF)]---END_TOOLS---".to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_fragmented_delimiter",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_json_repair() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"---TOOLS---[tool(MultiEdit, edits="""[{"old_string":"a","new_string":"b",}]""")]---END_TOOLS---"#.to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_json_repair",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_numeric_string_parsing() {
    let chunks = vec![OpenAIStreamChunk {
        id: "1".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                content: Some(
                    r#"---TOOLS---[tool(get_commit, sha="285e2cd")]---END_TOOLS---"#.to_string(),
                ),
                ..Default::default()
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        model: "google/gemini-2.5-pro".to_string(),
        usage: OpenAIUsage::default(),
    }];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_numeric_string_parsing",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_multibyte_character_parsing() {
    let chunks = vec![OpenAIStreamChunk {
        id: "1".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                content: Some(
                    r#"---TOOLS---[tool(TodoWrite, todos="""[{"content":"[INITIAL SURVEY] Read all documentation files (.md extension) in order: README.md →","status":"pending"}]""")]---END_TOOLS---"#.to_string(),
                ),
                ..Default::default()
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        model: "google/gemini-2.5-pro".to_string(),
        usage: OpenAIUsage::default(),
    }];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_utf8_character_parsing",
    )
    .await;
}

#[rstest]
fn parse_bracket_tool_name_with_hyphen_snapshot() {
    let slice = r#"[tool(mcp__mcp__searxng-search, query=\"rust\")]"#;
    let call = parse_bracket_tool(slice).expect("parse");
    insta::assert_debug_snapshot!("bracket_tools_parse_hyphenated_name", &call);
}

#[tokio::test]
async fn verify_bracket_tools_hyphen_tool_name_streaming() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---TOOLS---\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("[tool(mcp__mcp__searxng-search, query=\"ok\")]\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---END_TOOLS---".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_hyphenated_name_stream",
    )
    .await;
}

#[tokio::test]
async fn verify_xml_tools_html_inside_json_array() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("<function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#"<invoke name=\"MultiEdit\">"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#"<parameter name=\"edits\">```json
[{"old_string":"<div>hello</div>","new_string":"<b>world</b>"}]"#
                            .to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("</parameter></invoke></function_calls>".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-xml-tools",
        chunks,
        "xml_tools_html_inside_json_array",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_array_before_string() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---TOOLS---\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(
                        r#"[tool(MultiEdit, edits="""[{"old_string":"a","new_string":"b"}]""""#
                            .to_string(),
                    ),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(r#", file_path="""/app/main.rs""")]"#.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---END_TOOLS---".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "bracket_tools_array_before_string",
    )
    .await;
}

#[tokio::test]
async fn verify_bracket_tools_multiedit_execution() {
    let chunks = vec![
        OpenAIStreamChunk {
            id: "1".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("<end_cot>\n---TOOLS---\n".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "2".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("[tool(MultiEdit, file_path=\"\"\"".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "3".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("src/ue-bridge.go\"\"\", edits=\"\"\"[".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "4".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("{\"old_string\":\"foo\",\"new_string\":\"bar\"},".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "5".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("{\"old_string\":\"baz\",\"new_string\":\"qux\"}]".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "6".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("\"\"\")]".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
        OpenAIStreamChunk {
            id: "7".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("---END_TOOLS---".to_string()),
                    ..Default::default()
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "google/gemini-2.5-pro".to_string(),
            usage: OpenAIUsage::default(),
        },
    ];

    run_stream_conversion_test(
        "google/gemini-2.5-pro-bracket-tools",
        chunks,
        "multiedit_execution",
    )
    .await;
}
