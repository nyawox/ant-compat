use super::{
    parsing::{build_non_stream_response, parse_antml, parse_bracket_tools},
    streaming::{BracketGrammar, StreamingToolParser, XmlGrammar},
};
use crate::{
    adapters::traits::Adapter,
    conversion::request::Request,
    models::openai::{
        OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamFunction,
        OpenAIStreamToolCall,
    },
};
use async_stream::stream;
use futures_util::stream::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;

pub struct ToolSimulationResponseAdapter;

impl Adapter for ToolSimulationResponseAdapter {
    fn adapt_non_stream_response(&self, response: Value, request: &Request) -> Value {
        let mut response = response.clone();
        if let Some(text) = response["choices"][0]["message"]["content"].as_str() {
            let parsed = match request.model.as_str() {
                model if model.ends_with("-xml-tools") => parse_antml(text),
                model if model.ends_with("-bracket-tools") => parse_bracket_tools(text),
                _ => return response,
            };
            response["choices"][0]["message"] = build_non_stream_response(parsed);
        }
        response
    }

    fn adapt_chunk_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
        request: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
        let Some(mut parser) = initialize_parser(request) else {
            return stream;
        };

        Box::pin(stream! {
            let mut chunk_stream = stream;
            let mut last_chunk_for_metadata: Option<OpenAIStreamChunk> = None;

            while let Some(chunk_result) = chunk_stream.next().await {
                let mut chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(e);
                        continue;
                    }
                };

                let is_finish_chunk = chunk
                    .choices
                    .first()
                    .is_some_and(|c| c.finish_reason.is_some());

                if is_finish_chunk {
                    let final_tool_calls = parser.finalize();
                    if !final_tool_calls.is_empty()
                        && let Some(mut last_chunk) = last_chunk_for_metadata.take()
                    {
                        let start = parser.reserve_indices(final_tool_calls.len());
                        let tool_choice = build_tool_call_choice(
                            final_tool_calls,
                            start,
                            Some("tool_calls".to_string()),
                        );
                        last_chunk.choices = vec![tool_choice];
                        yield Ok(last_chunk);
                    }
                }

                if let Some(choice) = chunk.choices.first_mut()
                    && let Some(content) = choice.delta.content.take()
                {
                    let new_chunks = handle_delta(&content, &chunk, &mut parser);
                    for new_chunk in new_chunks {
                        yield Ok(new_chunk);
                    }
                }

                if !is_finish_chunk {
                    last_chunk_for_metadata = Some(chunk.clone());
                }

                yield Ok(chunk);
            }
        })
    }
}

fn initialize_parser(request: &Request) -> Option<StreamingToolParser> {
    match request.model.as_str() {
        model if model.ends_with("-xml-tools") => {
            Some(StreamingToolParser::new(Box::<XmlGrammar>::new(XmlGrammar)))
        }
        model if model.ends_with("-bracket-tools") => {
            Some(StreamingToolParser::new(Box::<BracketGrammar>::new(
                BracketGrammar,
            )))
        }
        _ => None,
    }
}

fn handle_delta(
    content: &str,
    chunk: &OpenAIStreamChunk,
    parser: &mut StreamingToolParser,
) -> Vec<OpenAIStreamChunk> {
    let mut chunks_to_yield = Vec::new();
    let (text_to_yield, tool_calls) = parser.process(content);

    if !text_to_yield.is_empty() {
        let mut text_chunk = chunk.clone();
        if let Some(text_choice) = text_chunk.choices.first_mut() {
            text_choice.delta.content = Some(text_to_yield);
            text_choice.finish_reason = None;
        }
        chunks_to_yield.push(text_chunk);
    }

    if !tool_calls.is_empty() {
        let start = parser.reserve_indices(tool_calls.len());
        let finish_reason = chunk.choices.first().and_then(|c| c.finish_reason.clone());
        let tool_choice = build_tool_call_choice(tool_calls, start, finish_reason);
        let mut tool_chunk = chunk.clone();
        tool_chunk.choices = vec![tool_choice];
        chunks_to_yield.push(tool_chunk);
    }

    chunks_to_yield
}

fn build_tool_call_choice(
    tool_calls: Vec<super::parsing::ParsedToolCall>,
    start_index: usize,
    finish_reason: Option<String>,
) -> OpenAIStreamChoice {
    OpenAIStreamChoice {
        index: 0,
        delta: OpenAIDelta {
            tool_calls: Some(
                tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tool)| OpenAIStreamToolCall {
                        index: u32::try_from(start_index.saturating_add(index)).unwrap_or(u32::MAX),
                        id: Some(format!("call_{}", rand::random::<u32>())),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIStreamFunction {
                            name: Some(tool.name),
                            arguments: Some(serde_json::to_string(&tool.args).unwrap_or_default()),
                        }),
                    })
                    .collect(),
            ),
            ..Default::default()
        },
        finish_reason,
    }
}
