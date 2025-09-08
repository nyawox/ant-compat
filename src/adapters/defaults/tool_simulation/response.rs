use super::{
    parsing::{build_non_stream_response, parse_antml, parse_bracket_tools},
    streaming::{BracketGrammar, StreamingToolParser, ToolEvent, XmlGrammar},
};
use crate::{
    adapters::traits::Adapter,
    conversion::request::Request,
    error::AppError,
    models::openai::{
        OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamFunction,
        OpenAIStreamToolCall,
    },
};
use async_stream::stream;
use futures_util::stream::{Stream, StreamExt};
use serde_json::Value;
use std::{collections::HashMap, pin::Pin};

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
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>>,
        request: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>> {
        let Some(mut parser) = initialize_parser(request) else {
            return stream;
        };

        Box::pin(stream! {
            let mut chunk_stream = stream;
            let mut last_chunk_for_metadata: Option<OpenAIStreamChunk> = None;
            let mut call_ids: HashMap<usize, String> = HashMap::new();

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
                    let final_events = parser.finalize().await;
                    if !final_events.is_empty()
                        && let Some(last_chunk) = last_chunk_for_metadata.clone()
                    {
                        for event in final_events {
                            if let Some(choice) = tool_event_to_choice(
                                event,
                                &mut call_ids,
                            ) {
                                let mut out = last_chunk.clone();
                                out.choices = vec![choice];
                                yield Ok(out);
                            }
                        }
                    }
                }

                if let Some(choice) = chunk.choices.first_mut() {
                    let (content_opt, reasoning_opt) = {
                        let delta = &mut choice.delta;
                        (delta.content.take(), delta.reasoning_content.take())
                    };

                    let deltas = [
                        content_opt.map(|text| (text, false)),
                        reasoning_opt.map(|text| (text, true)),
                    ];

                    let mut tools_block_end = false;
                    for (text, is_reasoning) in deltas.into_iter().flatten() {
                        let (produced, tools_block_end_detected) = handle_delta(&text, &chunk, &mut parser, is_reasoning, &mut call_ids).await;
                        for new_chunk in produced {
                            yield Ok(new_chunk);
                        }
                        if tools_block_end_detected {
                            tools_block_end = true;
                        }
                    }

                    if tools_block_end && !is_finish_chunk {
                        let mut finish_chunk = chunk.clone();
                        finish_chunk.choices = vec![OpenAIStreamChoice {
                            index: 0,
                            delta: OpenAIDelta::default(),
                            finish_reason: Some("tool_calls".to_string()),
                        }];
                        yield Ok(finish_chunk);
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

async fn handle_delta(
    content: &str,
    chunk: &OpenAIStreamChunk,
    parser: &mut StreamingToolParser,
    is_reasoning: bool,
    call_ids: &mut HashMap<usize, String>,
) -> (Vec<OpenAIStreamChunk>, bool) {
    let mut chunks_to_yield = Vec::new();
    let (text_to_yield, events) = parser.process(content).await;

    if !text_to_yield.is_empty() {
        let mut text_chunk = chunk.clone();
        if let Some(text_choice) = text_chunk.choices.first_mut() {
            if is_reasoning {
                text_choice.delta.reasoning_content = Some(text_to_yield);
            } else {
                text_choice.delta.content = Some(text_to_yield);
            }
            text_choice.finish_reason = None;
        }
        chunks_to_yield.push(text_chunk);
    }

    let mut tools_block_end = false;
    for event in events {
        match event {
            ToolEvent::ToolsBlockEnd => {
                tools_block_end = true;
            }
            other => {
                if let Some(choice) = tool_event_to_choice(other, call_ids) {
                    let mut tool_chunk = chunk.clone();
                    tool_chunk.choices = vec![choice];
                    chunks_to_yield.push(tool_chunk);
                }
            }
        }
    }

    (chunks_to_yield, tools_block_end)
}

fn tool_event_to_choice(
    event: ToolEvent,
    call_ids: &mut HashMap<usize, String>,
) -> Option<OpenAIStreamChoice> {
    match event {
        ToolEvent::Start { index, name } => {
            let id = call_ids
                .entry(index)
                .or_insert_with(|| format!("call_{}", rand::random::<u32>()));
            Some(OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    tool_calls: Some(vec![OpenAIStreamToolCall {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        id: Some(id.clone()),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIStreamFunction {
                            name: Some(name),
                            arguments: None,
                        }),
                    }]),
                    ..Default::default()
                },
                finish_reason: None,
            })
        }
        ToolEvent::Arg { index, delta } => {
            let id = call_ids
                .entry(index)
                .or_insert_with(|| format!("call_{}", rand::random::<u32>()));
            Some(OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    tool_calls: Some(vec![OpenAIStreamToolCall {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        id: Some(id.clone()),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIStreamFunction {
                            name: None,
                            arguments: Some(delta),
                        }),
                    }]),
                    ..Default::default()
                },
                finish_reason: None,
            })
        }
        ToolEvent::End | ToolEvent::ToolsBlockEnd => None,
    }
}
