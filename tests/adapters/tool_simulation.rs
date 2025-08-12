use ant_compat::{
    adapters::RequestAdapter,
    conversion::{request::convert_claude_to_openai, stream::convert_openai_stream_to_anthropic},
    models::{
        claude::{
            AnthropicStreamEvent, ClaudeContent, ClaudeContentBlock, ClaudeMessage,
            ClaudeMessagesRequest, ClaudeSystem, ClaudeTool, ClaudeToolChoice, ContentBlock,
        },
        openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIUsage},
    },
};
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use rstest::rstest;
use serde_json::{Value, json};

async fn mock_response_from_chunks(chunks: Vec<OpenAIStreamChunk>) -> reqwest::Response {
    let sse_data: Vec<u8> = chunks
        .into_iter()
        .map(|chunk| {
            let json = serde_json::to_string(&chunk)
                .expect("Serialization of a test data struct should not fail");
            format!("data: {json}\n\n")
        })
        .collect::<String>()
        .into_bytes();

    let body = reqwest::Body::from(sse_data);
    let response = http::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(body)
        .expect("Building a static HTTP response should not fail");

    reqwest::Response::from(response)
}

async fn collect_and_parse_stream(
    stream: impl Stream<Item = Result<Bytes, std::io::Error>>,
) -> Vec<AnthropicStreamEvent> {
    let mut events = Vec::new();
    let mut stream = Box::pin(stream);

    while let Some(item) = stream.next().await {
        let bytes = item.expect("Test stream should not produce I/O errors");
        let lines = String::from_utf8_lossy(&bytes);
        for line in lines.split('\n') {
            if line.starts_with("data: ") {
                let json_str = &line[6..];
                if let Ok(event) = serde_json::from_str(json_str) {
                    events.push(event);
                }
            }
        }
    }
    events
}

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
    let adapter = RequestAdapter::for_model(model);
    let result = convert_claude_to_openai(request, model, &adapter);
    insta::assert_debug_snapshot!(snapshot_name, &result);
}

async fn run_stream_conversion_test(
    model: &str,
    chunks: Vec<OpenAIStreamChunk>,
    snapshot_name: &str,
) {
    let request = get_request(model);
    let mock_response = mock_response_from_chunks(chunks).await;
    let adapter = RequestAdapter::for_model(model);

    let anthropic_stream =
        convert_openai_stream_to_anthropic(mock_response, model, &adapter, &request);

    let mut events = collect_and_parse_stream(anthropic_stream).await;
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
                        r#"[tool(Test, json="""{"key": "value with quotes"}""")]"#.to_string(),
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
                    content: Some(r#"[tool(NoArgs)]"#.to_string()),
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
                    content: Some(r#"[tool(WithParens, arg="(value)")]"#.to_string()),
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
