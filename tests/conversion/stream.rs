use super::scenarios::{final_chunk, text_chunk, thinking_chunk, tool_chunk, tool_chunk_partial};
use crate::helpers;
use ant_compat::{
    adapters::RequestAdapter,
    conversion::stream::convert_openai_stream_to_anthropic,
    directives::models::Settings,
    models::{
        claude::{AnthropicStreamEvent, ClaudeMessagesRequest, MessageStart},
        openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIUsage},
    },
};
use rstest::rstest;

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
#[case("think_in_text", vec![text_chunk("Foo <think>bar</think> baz"), final_chunk("stop")])]
#[case("think_in_text_reentry", vec![text_chunk("Foo <think>a</think> x <think>b</think> y"), final_chunk("stop")])]
#[case("nested_thinks_in_text", vec![text_chunk("Foo <think><think><think><think><think>a</think></think></think></think> x <think><think><think><think><think><think><think><think><think><think>b</think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think> y"), final_chunk("stop")])]
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
    let mock_response = crate::helpers::mock_response_from_chunks(chunks).await;
    let model = "test-model";
    let request = ClaudeMessagesRequest {
        model: model.to_string(),
        messages: vec![],
        max_tokens: 1024,
        stream: Some(true),
        system: None,
        stop_sequences: None,
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };
    let adapter = RequestAdapter::for_model(model, &Settings::default());
    let mock_state = helpers::mock_app_state();

    let anthropic_stream =
        convert_openai_stream_to_anthropic(mock_response, model, &adapter, &request, &mock_state);

    let mut output_events = crate::helpers::collect_and_parse_stream(anthropic_stream).await;
    redact_message_ids(&mut output_events);

    insta::assert_debug_snapshot!(name, output_events);
}
