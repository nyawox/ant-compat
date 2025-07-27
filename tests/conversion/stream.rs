use super::scenarios::{final_chunk, text_chunk, thinking_chunk, tool_chunk, tool_chunk_partial};
use ant_compat::conversion::stream::convert_openai_stream_to_anthropic;
use ant_compat::models::claude::{AnthropicStreamEvent, MessageStart};
use ant_compat::models::openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIUsage};
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use rstest::rstest;

async fn mock_response_from_chunks(chunks: Vec<OpenAIStreamChunk>) -> reqwest::Response {
    let sse_data: Vec<u8> = chunks
        .into_iter()
        .map(|chunk| {
            let json = serde_json::to_string(&chunk)
                .expect("Serialization of a test data struct should not fail");
            format!("data: {}\n\n", json)
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

fn redact_message_ids(events: &mut [AnthropicStreamEvent]) {
    for event in events {
        if let AnthropicStreamEvent::MessageStart(MessageStart { message, .. }) = event {
            message.id = "[redacted-id]".to_string();
        }
    }
}

#[rstest]
#[case("simple_text", vec![text_chunk("Hello"), final_chunk("stop")])]
#[case("fragmented_text", vec![text_chunk("Hel"), text_chunk("lo"), final_chunk("stop")])]
#[case("single_tool", vec![tool_chunk(0, "id1", "get_weather", r#"{"loc":"SF"}"#), final_chunk("tool_calls")])]
#[case("fragmented_args", vec![tool_chunk(0, "id1", "search", r#"{"q":"#), tool_chunk(0, "id1", "search", r#"rust"}"#), final_chunk("tool_calls")])]
#[case("text_then_tool", vec![text_chunk("Yapping..."), tool_chunk(1, "id1", "search", "{}"), final_chunk("tool_calls")])]
#[case("tool_then_text", vec![tool_chunk(0, "id1", "search", "{}"), text_chunk("Found results."), final_chunk("stop")])]
#[case("thinking_then_text", vec![thinking_chunk("Thinking..."), text_chunk("Okay"), final_chunk("stop")])]
#[case("multiple_choices", vec![
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![
            OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some("A".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            },
            OpenAIStreamChoice {
                index: 1,
                delta: OpenAIDelta {
                    content: Some("B".to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            },
        ],
        model: "gpt-4.1".to_string(),
        usage: OpenAIUsage::default(),
    },
    final_chunk("stop"),
])]
#[case("early_finish", vec![tool_chunk(0, "id1", "part1", "{"), final_chunk("tool_calls"), tool_chunk(0, "id1", "part1", "}")])]
#[case(
    "fragmented_tools",
    vec![
        tool_chunk_partial(0, Some("id1"), None, None),
        tool_chunk_partial(0, None, Some("get_weather"), None),
        tool_chunk_partial(0, None, None, Some(r#"{"location":"SF"}"#)),
        final_chunk("tool_calls"),
    ]
)]
#[case(
    "parallel_tools",
    vec![
        tool_chunk(0, "id1", "tool1", r#"{"arg":"value1"}"#),
        tool_chunk(1, "id2", "tool2", r#"{"arg":"value2"}"#),
        tool_chunk(2, "id3", "tool3", r#"{"arg":"value3"}"#),
        tool_chunk(3, "id4", "tool4", r#"{"arg":"value4"}"#),
        tool_chunk(4, "id5", "tool5", r#"{"arg":"value5"}"#),
        final_chunk("tool_calls"),
    ]
)]
#[tokio::test]
async fn verify_stream_conversion(#[case] name: &str, #[case] chunks: Vec<OpenAIStreamChunk>) {
    let mock_response = mock_response_from_chunks(chunks).await;

    let anthropic_stream =
        convert_openai_stream_to_anthropic(mock_response, "test-model".to_string());

    let mut output_events = collect_and_parse_stream(anthropic_stream).await;
    redact_message_ids(&mut output_events);

    insta::assert_debug_snapshot!(name, output_events);
}
