use async_stream::stream;
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt, TryStreamExt};
use serde_json::{Map, Value, json};
use std::{
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_sse_codec::{Frame, SseDecoder};
use tokio_util::codec::FramedRead;
use tokio_util::io::StreamReader;
use tracing::{debug, error};

use crate::{
    adapters::RequestAdapter,
    error::AppError,
    models::{
        claude::{
            AnthropicStreamEvent, ClaudeMessagesRequest, ClaudeStreamMessage, ClaudeStreamUsage,
            ContentBlock, ContentBlockDelta, ContentBlockStart, ContentBlockStop, Delta,
            FinishReason, MessageDelta, MessageDeltaInfo, MessageStart, MessageStop,
        },
        openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamToolCall},
        shared::{
            ActiveState, NextState, StreamState, decide_after_reasoning, decide_after_text,
            decide_after_tool, decide_next_state,
        },
    },
    state::AppState,
};

pub fn emit_event(event_type: &str, data: &impl serde::Serialize) -> Bytes {
    let data_str = serde_json::to_string(data).unwrap_or_default();
    debug!("Emitting event: {event_type}");
    Bytes::from(format!("event: {event_type}\ndata: {data_str}\n\n"))
}

pub fn emit_ping() -> Bytes {
    debug!("Emitting ping event");
    emit_event("ping", &json!({"type": "ping"}))
}

pub fn emit_initial_events(state: &StreamState) -> Vec<AnthropicStreamEvent> {
    debug!("Emitting initial events for message: {}", state.message_id);
    vec![AnthropicStreamEvent::MessageStart(MessageStart {
        message: ClaudeStreamMessage {
            id: state.message_id.clone(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: state.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: ClaudeStreamUsage {
                input_tokens: 0,
                output_tokens: 0,
            },
        },
    })]
}

#[must_use]
pub fn emit_final_events(
    current_state: ActiveState,
    context: &StreamState,
    finish_reason: &str,
) -> Vec<AnthropicStreamEvent> {
    let mut events = Vec::new();

    if let Some(content_index) = current_state.content_index() {
        events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
            index: content_index,
        }));
    }

    if let Some(tool_index) = context.tool_index
        && let Some(tool_call) = context.tool_calls.get(&tool_index)
        && let Some(content_index) = tool_call.content_index
    {
        events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
            index: content_index,
        }));
    }

    let stop_reason = if context.tool_calls.is_empty() {
        FinishReason(Some(finish_reason))
            .to_anthropic_stop_reason()
            .to_string()
    } else {
        // workaround cases where upstream apis sends another ContentBlockStop with end_turn after tool use
        // but the client expects a tool_use stop reason
        "tool_use".to_string()
    };
    events.push(AnthropicStreamEvent::MessageDelta(MessageDelta {
        delta: MessageDeltaInfo {
            stop_reason,
            stop_sequence: None,
        },
        usage: context.usage_data.clone(),
    }));

    events.push(AnthropicStreamEvent::MessageStop(MessageStop {}));

    events
}

fn generate_unique_id(prefix: &str, index: u32) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}_{timestamp}_{index}")
}

pub fn update_usage_from_chunk(chunk: &OpenAIStreamChunk, state: &mut StreamState) {
    let usage = &chunk.usage;
    state.usage_data.input = usage.prompt_tokens;
    state.usage_data.output = usage.completion_tokens;
    if let Some(details) = &usage.prompt_tokens_details
        && let Some(cached_tokens) = details.cached_tokens
    {
        state.usage_data.cache_read_input = Some(cached_tokens);
    }
    if state.usage_data.input != 0 || state.usage_data.output != 0 {
        debug!(
            "Updated usage data: input={}, output={}",
            state.usage_data.input, state.usage_data.output
        );
    }
}

pub fn handle_tool_calls_delta(
    tool_calls_delta: &[OpenAIStreamToolCall],
    state: &mut StreamState,
) -> Vec<AnthropicStreamEvent> {
    debug!("Handling tool calls delta: {tool_calls_delta:?}");
    tool_calls_delta
        .iter()
        .flat_map(|tool_call| process_tool_call_delta(tool_call, state))
        .collect()
}

fn process_tool_call_delta(
    tool_call: &OpenAIStreamToolCall,
    state: &mut StreamState,
) -> Vec<AnthropicStreamEvent> {
    let mut events = Vec::new();

    let has_function_name = tool_call
        .function
        .as_ref()
        .and_then(|function| function.name.as_ref())
        .is_some();

    let entry_has_name = state
        .tool_calls
        .get(&tool_call.index)
        .and_then(|existing| existing.name.as_ref())
        .is_some();

    let starts_new_tool_use = has_function_name && !entry_has_name;

    if starts_new_tool_use {
        if let Some(stop_event) = state
            .tool_index
            .filter(|&active_index| active_index != tool_call.index)
            .and_then(|active_index| state.tool_calls.get(&active_index))
            .and_then(|active_tool| active_tool.content_index)
            .map(|content_index| {
                AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                })
            })
        {
            events.push(stop_event);
        }
        state.tool_index = Some(tool_call.index);
    }

    let entry = state.tool_calls.entry(tool_call.index).or_default();

    if let Some(id) = &tool_call.id {
        entry.id = Some(id.clone());
    }

    if let Some(function) = &tool_call.function {
        if let (Some(name), None) = (function.name.as_ref(), entry.name.as_ref()) {
            entry.name = Some(name.clone());

            let content_index = state.next_content_index;
            state.next_content_index += 1;
            entry.content_index = Some(content_index);

            let tool_use_id = entry
                .id
                .clone()
                .unwrap_or_else(|| generate_unique_id("call", tool_call.index));

            events.push(AnthropicStreamEvent::ContentBlockStart(ContentBlockStart {
                index: content_index,
                content_block: ContentBlock::ToolUse {
                    id: tool_use_id,
                    name: name.clone(),
                    input: Value::Object(Map::new()),
                },
            }));
        }

        if let Some(arguments) = &function.arguments
            && !arguments.is_empty()
        {
            entry.arguments.push_str(arguments);
            if let Some(content_index) = entry.content_index {
                events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::InputJson {
                        partial_json: arguments.clone(),
                    },
                }));
            }
        }
    }

    events
}

#[must_use]
pub fn convert_openai_stream_to_anthropic(
    response: reqwest::Response,
    model: &str,
    adapter: &RequestAdapter,
    request: &ClaudeMessagesRequest,
    state: &AppState,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, AppError>> + Send>> {
    let byte_stream = response.bytes_stream().map_err(AppError::from);

    let stream_reader = StreamReader::new(byte_stream);

    let chunk_stream =
        FramedRead::new(stream_reader, SseDecoder::<String>::new()).filter_map(|frame| async {
            match frame {
                Ok(Frame::Event(event)) => {
                    let data = &event.data;
                    if data == "[DONE]" {
                        return None;
                    }
                    match serde_json::from_str::<OpenAIStreamChunk>(data) {
                        Ok(chunk) => Some(Ok(chunk)),
                        Err(e) => Some(Err(AppError::StreamError(format!(
                            "Failed to parse stream chunk: {e}, data: {data}"
                        )))),
                    }
                }
                Ok(_) => None,
                Err(e) => Some(Err(e.into())),
            }
        });

    let adapted_chunk_stream = adapter.adapt_chunk_stream(Box::pin(chunk_stream), request);
    let event_stream = chunks_to_events(model, adapted_chunk_stream, state.idle_connection_timeout);

    Box::pin(stream! {
        let mut stream = Box::pin(event_stream);
        let mut ping_interval = tokio::time::interval(std::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                event_result = stream.next() => {
                    if let Some(event_result) = event_result {
                        match event_result {
                            Ok(event) => {
                                let (event_type, data) = event.to_parts();
                                yield Ok(emit_event(event_type, &data));
                            }
                            Err(e) => {
                                let error_event = json!({
                                    "type": "error",
                                    "error": { "type": "api_error", "message": e.to_string() }
                                });
                                yield Ok(emit_event("error", &error_event));
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                }
                _ = ping_interval.tick() => {
                    yield Ok(emit_ping());
                }
            }
        }
    })
}

#[must_use]
pub fn chunks_to_events(
    model: &str,
    mut chunk_stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>>,
    idle_timeout_secs: u64,
) -> Pin<Box<dyn Stream<Item = Result<AnthropicStreamEvent, AppError>> + Send>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let message_id = format!("msg_{timestamp}");

    let mut state = StreamState {
        model: model.to_string(),
        message_id,
        ..Default::default()
    };

    Box::pin(stream! {
        for initial_event in emit_initial_events(&state) {
            yield Ok(initial_event);
        }

        let mut stream = chunk_stream.as_mut();
        let mut last_state = state.state;
        let timeout_duration = Duration::from_secs(idle_timeout_secs);
        loop {
            let chunk_result = match tokio::time::timeout(timeout_duration, stream.next()).await {
                Ok(Some(result)) => result,
                Ok(None) => break,
                Err(_) => {
                    error!("Stream timed out after {idle_timeout_secs} seconds of inactivity");
                    break;
                }
            };
            let chunk = match chunk_result {
                Ok(chunk_value) => chunk_value,
                Err(error) => {
                    yield Err(error);
                    break;
                }
            };

            update_usage_from_chunk(&chunk, &mut state);

            if state.finish_reason.is_some() {
                continue;
            }

            for choice in chunk.choices {
                for event in state.process_choice(&choice) {
                    yield Ok(event);
                }
            }
            last_state = state.state;
        }

        if let Some(reason) = &state.finish_reason {
            for event in emit_final_events(last_state, &state, reason) {
                yield Ok(event);
            }
        } else {
            for event in emit_final_events(last_state, &state, "stop_sequence") {
                yield Ok(event);
            }
        }
    })
}

impl StreamState {
    pub(crate) fn process_choice(
        &mut self,
        choice: &OpenAIStreamChoice,
    ) -> Vec<AnthropicStreamEvent> {
        let use_preprocess = !matches!(self.state, ActiveState::Tool);
        let prepared_choice = if use_preprocess {
            match choice.delta.content.as_deref() {
                Some(content) => {
                    let processed = self.think_parser.preprocess(content);
                    OpenAIStreamChoice {
                        index: choice.index,
                        delta: OpenAIDelta {
                            content: Some(processed),
                            ..choice.delta.clone()
                        },
                        finish_reason: choice.finish_reason.clone(),
                    }
                }
                None => choice.clone(),
            }
        } else {
            choice.clone()
        };
        let current_state = std::mem::take(&mut self.state);
        let (new_state, events) = Self::transition(current_state, &prepared_choice, self);
        self.state = new_state;
        events
    }

    fn transition(
        current_state: ActiveState,
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        match current_state {
            ActiveState::Idle => Self::handle_idle(choice, context),
            ActiveState::Thinking {
                content_index,
                via_think_tag,
            } => {
                if via_think_tag {
                    Self::handle_thinking(choice, content_index, context)
                } else {
                    Self::handle_reasoning(choice, content_index, context)
                }
            }
            ActiveState::Text { content_index } => {
                Self::handle_text(choice, content_index, context)
            }
            ActiveState::Tool => Self::handle_tool(choice, context),
        }
    }

    fn handle_idle(
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        match decide_next_state(choice, &context.think_parser) {
            NextState::Tool(tool_calls) => (
                ActiveState::Tool,
                handle_tool_calls_delta(tool_calls, context),
            ),
            NextState::Finish(finish_reason) => {
                context.finish_reason = Some(finish_reason.clone());
                (ActiveState::Idle, vec![])
            }
            NextState::Think { via_think_tag } => {
                Self::start_thinking(choice, context, via_think_tag)
            }
            NextState::Text => {
                let index = context.next_content_index;
                context.next_content_index += 1;
                let (mut new_state, mut events) = Self::start_text(index, context);
                if let Some(content_index) = new_state.content_index() {
                    let (text_state, text_events) =
                        Self::handle_text(choice, content_index, context);
                    new_state = text_state;
                    events.extend(text_events);
                }
                (new_state, events)
            }
            NextState::Idle => (ActiveState::Idle, vec![]),
        }
    }

    fn start_thinking(
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
        via_think_tag: bool,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        let index = context.next_content_index;
        context.next_content_index += 1;
        let mut events = vec![AnthropicStreamEvent::ContentBlockStart(ContentBlockStart {
            index,
            content_block: ContentBlock::Thinking {
                thinking: String::new(),
            },
        })];

        let (new_state, new_events) = if via_think_tag {
            let content_to_process = context
                .think_parser
                .clean_before(choice.delta.content.as_deref().unwrap_or(""));

            let cleaned_choice = OpenAIStreamChoice {
                index: choice.index,
                delta: OpenAIDelta {
                    content: Some(content_to_process),
                    ..choice.delta.clone()
                },
                finish_reason: choice.finish_reason.clone(),
            };
            Self::handle_thinking(&cleaned_choice, index, context)
        } else {
            context.think_parser.on_reasoning_mode();
            Self::handle_reasoning(choice, index, context)
        };

        events.extend(new_events);
        (new_state, events)
    }

    fn handle_thinking(
        choice: &OpenAIStreamChoice,
        content_index: u32,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
            let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                index: content_index,
            })];
            events.extend(handle_tool_calls_delta(tool_calls, context));
            return (ActiveState::Tool, events);
        }

        let content_str = choice.delta.content.as_deref().unwrap_or("");

        if choice.delta.has_think_end_tag()
            && let Some(end_tag_pos) = content_str
                .find("</think>")
                .or_else(|| content_str.find("</cot>"))
                .or_else(|| content_str.find("<end_cot>"))
        {
            let mut events = Vec::new();
            let final_thinking = &content_str[..end_tag_pos];

            if !final_thinking.is_empty() {
                events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::Thinking {
                        thinking: final_thinking.to_string(),
                    },
                }));
            }

            events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                index: content_index,
            }));

            let remaining_text = context
                .think_parser
                .clean_after(&content_str[end_tag_pos..]);

            context.think_parser.on_think_end();

            if !remaining_text.is_empty() {
                let remaining_choice = OpenAIStreamChoice {
                    index: choice.index,
                    delta: OpenAIDelta {
                        content: Some(remaining_text.to_string()),
                        ..choice.delta.clone()
                    },
                    finish_reason: choice.finish_reason.clone(),
                };
                let (new_state, remaining_events) = Self::handle_idle(&remaining_choice, context);
                events.extend(remaining_events);
                return (new_state, events);
            }
            return (ActiveState::Idle, events);
        }

        if content_str.is_empty() {
            (
                ActiveState::Thinking {
                    content_index,
                    via_think_tag: true,
                },
                vec![],
            )
        } else {
            let event = AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                index: content_index,
                delta: Delta::Thinking {
                    thinking: content_str.to_string(),
                },
            });
            (
                ActiveState::Thinking {
                    content_index,
                    via_think_tag: true,
                },
                vec![event],
            )
        }
    }

    fn handle_reasoning(
        choice: &OpenAIStreamChoice,
        content_index: u32,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        match decide_after_reasoning(choice) {
            NextState::Tool(tool_calls) => {
                let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                })];
                events.extend(handle_tool_calls_delta(tool_calls, context));
                (ActiveState::Tool, events)
            }
            NextState::Finish(finish_reason) => {
                context.finish_reason = Some(finish_reason.clone());
                (
                    ActiveState::Thinking {
                        content_index,
                        via_think_tag: false,
                    },
                    vec![],
                )
            }
            NextState::Text => {
                let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                })];
                let (mut new_state, start_text_events) =
                    Self::start_text(context.next_content_index, context);
                events.extend(start_text_events);
                let (text_state, text_events) = Self::handle_text(
                    choice,
                    new_state.content_index().unwrap_or(content_index),
                    context,
                );
                new_state = text_state;
                events.extend(text_events);
                (new_state, events)
            }
            NextState::Think { .. } => {
                let mut events = vec![];
                if let Some(reasoning) = choice.delta.get_reasoning() {
                    events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                        index: content_index,
                        delta: Delta::Thinking {
                            thinking: reasoning.clone(),
                        },
                    }));
                }
                (
                    ActiveState::Thinking {
                        content_index,
                        via_think_tag: false,
                    },
                    events,
                )
            }
            NextState::Idle => (
                ActiveState::Thinking {
                    content_index,
                    via_think_tag: false,
                },
                vec![],
            ),
        }
    }

    fn start_text(
        index: u32,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        context.next_content_index = index + 1;
        let event = AnthropicStreamEvent::ContentBlockStart(ContentBlockStart {
            index,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        });
        (
            ActiveState::Text {
                content_index: index,
            },
            vec![event],
        )
    }

    fn handle_text(
        choice: &OpenAIStreamChoice,
        content_index: u32,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        if context.think_parser.is_thinking_allowed()
            && choice.delta.has_think_tag()
            && let Some(content) = choice.delta.content.as_deref()
            && let Some(think_start_pos) = content.find("<think>").or_else(|| content.find("<cot>"))
        {
            let mut events = Vec::new();
            let preceding_text = &content[..think_start_pos];
            let remaining_chunk = context
                .think_parser
                .clean_before(&content[think_start_pos..]);

            if !preceding_text.is_empty() {
                events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::Text {
                        text: preceding_text.to_string(),
                    },
                }));
            }

            events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                index: content_index,
            }));

            let remaining_choice = OpenAIStreamChoice {
                index: choice.index,
                delta: OpenAIDelta {
                    content: Some(remaining_chunk.to_string()),
                    ..choice.delta.clone()
                },
                finish_reason: choice.finish_reason.clone(),
            };

            let (new_state, remaining_events) = Self::handle_idle(&remaining_choice, context);
            events.extend(remaining_events);
            return (new_state, events);
        }

        match decide_after_text(choice) {
            NextState::Tool(tool_calls) => {
                let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                })];
                events.extend(handle_tool_calls_delta(tool_calls, context));
                (ActiveState::Tool, events)
            }
            NextState::Finish(finish_reason) => {
                context.finish_reason = Some(finish_reason.clone());
                (ActiveState::Text { content_index }, vec![])
            }
            NextState::Text => {
                let event = AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::Text {
                        text: choice.delta.content.clone().unwrap_or_default(),
                    },
                });
                (ActiveState::Text { content_index }, vec![event])
            }
            NextState::Idle | NextState::Think { .. } => {
                (ActiveState::Text { content_index }, vec![])
            }
        }
    }

    fn handle_tool(
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        match decide_after_tool(choice) {
            NextState::Finish(finish_reason) => {
                context.finish_reason = Some(finish_reason.clone());
                let mut events = vec![];
                if let Some(active_index) = context.tool_index
                    && let Some(active_tool) = context.tool_calls.get_mut(&active_index)
                    && let Some(content_index) = active_tool.content_index.take()
                {
                    events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                        index: content_index,
                    }));
                }
                context.tool_index = None;
                (ActiveState::Tool, events)
            }
            NextState::Tool(tool_calls) => (
                ActiveState::Tool,
                handle_tool_calls_delta(tool_calls, context),
            ),
            // prevent premature declaration of hallucinated tool success
            // models are traind to wait for and react to tool feedback
            // even during reasoning phase
            NextState::Text | NextState::Think { .. } | NextState::Idle => {
                (ActiveState::Tool, vec![])
            }
        }
    }
}
