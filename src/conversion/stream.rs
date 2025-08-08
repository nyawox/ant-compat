use async_stream::stream;
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use serde_json::{Map, Value, json};
use std::{
    pin::Pin,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::{debug, error};

use crate::{
    adapters::RequestAdapter,
    models::{
        claude::{
            AnthropicStreamEvent, ClaudeMessagesRequest, ClaudeStreamMessage, ClaudeStreamUsage,
            ContentBlock, ContentBlockDelta, ContentBlockStart, ContentBlockStop, Delta,
            FinishReason, MessageDelta, MessageDeltaInfo, MessageStart, MessageStop,
        },
        openai::{OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamToolCall},
        shared::{ActiveState, StreamState},
    },
};

pub fn emit_event(event_type: &str, data: &impl serde::Serialize) -> Bytes {
    let data_str = serde_json::to_string(data).unwrap_or_default();
    debug!("Emitting event: {event_type}, data: {data_str}");
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
    debug!(
        "Updated usage data: input={}, output={}",
        state.usage_data.input, state.usage_data.output
    );
}

pub fn handle_tool_calls_delta(
    tool_calls_delta: &[OpenAIStreamToolCall],
    state: &mut StreamState,
) -> Vec<AnthropicStreamEvent> {
    let mut events = Vec::new();
    debug!("Handling tool calls delta: {tool_calls_delta:?}");

    if tool_calls_delta.is_empty() {
        return events;
    }

    for tool_call in tool_calls_delta {
        let is_new_tool = tool_call
            .function
            .as_ref()
            .and_then(|func| func.name.as_ref())
            .is_some()
            && state
                .tool_calls
                .get(&tool_call.index)
                .is_none_or(|tool| tool.name.is_none());

        if is_new_tool {
            if let Some(active_index) = state.tool_index
                && active_index != tool_call.index
                && let Some(last_tool) = state.tool_calls.get(&active_index)
                && let Some(content_index) = last_tool.content_index
            {
                events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                }));
            }
            state.tool_index = Some(tool_call.index);
        }

        let entry = state.tool_calls.entry(tool_call.index).or_default();

        if let Some(id) = &tool_call.id {
            entry.id = Some(id.clone());
        }

        if let Some(function) = &tool_call.function {
            if let Some(name) = &function.name
                && entry.name.is_none()
            {
                entry.name = Some(name.clone());
                let content_index = state.next_content_index;
                state.next_content_index += 1;
                entry.content_index = Some(content_index);

                events.push(AnthropicStreamEvent::ContentBlockStart(ContentBlockStart {
                    index: content_index,
                    content_block: ContentBlock::ToolUse {
                        id: entry
                            .id
                            .clone()
                            .unwrap_or_else(|| generate_unique_id("call", tool_call.index)),
                        name: name.clone(),
                        input: Value::Object(Map::new()),
                    },
                }));
            }

            if let Some(arguments) = &function.arguments {
                entry.arguments.push_str(arguments);

                if !arguments.is_empty()
                    && let Some(content_index) = entry.content_index
                {
                    events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                        index: content_index,
                        delta: Delta::InputJson {
                            partial_json: arguments.clone(),
                        },
                    }));
                }
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
) -> Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    let byte_stream = response.bytes_stream();
    let chunk_stream = bytes_to_chunks(Box::pin(byte_stream));
    let adapted_chunk_stream = adapter.adapt_chunk_stream(Box::pin(chunk_stream), request);
    let event_stream = chunks_to_events(model, adapted_chunk_stream);

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
                                yield Err(std::io::Error::other(e));
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

fn bytes_to_chunks(
    mut byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
    Box::pin(stream! {
        let mut stream_of_bytes = byte_stream.as_mut();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream_of_bytes.next().await {
            let chunk_bytes = match chunk_result {
                Ok(bytes) => bytes,
                Err(e) => { yield Err(e); break; }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));
            let mut lines: Vec<&str> = buffer.split('\n').collect();
            let new_buffer = if buffer.ends_with('\n') { String::new() } else { lines.pop().unwrap_or("").to_string() };

            for line in lines {
                if !line.starts_with("data: ") { continue; }
                let data = &line[6..];
                if data == "[DONE]" { break; }
                match serde_json::from_str::<OpenAIStreamChunk>(data) {
                    Ok(chunk) => yield Ok(chunk),
                    Err(e) => { error!("Failed to parse stream chunk: {e}, data: {data}"); }
                }
            }
            buffer = new_buffer;
        }
    })
}

#[must_use]
pub fn chunks_to_events(
    model: &str,
    mut chunk_stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
) -> Pin<Box<dyn Stream<Item = Result<AnthropicStreamEvent, reqwest::Error>> + Send>> {
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
        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
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
            error!("Stream ended prematurely without a finish reason");
        }
    })
}

impl StreamState {
    pub(crate) fn process_choice(
        &mut self,
        choice: &OpenAIStreamChoice,
    ) -> Vec<AnthropicStreamEvent> {
        let current_state = std::mem::take(&mut self.state);
        let (new_state, events) = Self::transition(current_state, choice, self);
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
            ActiveState::Thinking { content_index } => {
                Self::handle_thinking(choice, content_index, context)
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
        if let Some(tool_calls) = &choice.delta.tool_calls {
            let tool_events = handle_tool_calls_delta(tool_calls, context);
            return (ActiveState::Tool, tool_events);
        }

        if let Some(finish_reason) = &choice.finish_reason {
            context.finish_reason = Some(finish_reason.clone());
            return (ActiveState::Idle, vec![]);
        }

        if choice.delta.thinking.is_some() || choice.delta.reasoning.is_some() {
            let index = context.next_content_index;
            context.next_content_index += 1;
            let mut events = vec![AnthropicStreamEvent::ContentBlockStart(ContentBlockStart {
                index,
                content_block: ContentBlock::Thinking {
                    thinking: String::new(),
                },
            })];
            let (new_state, new_events) = Self::handle_thinking(choice, index, context);
            events.extend(new_events);
            return (new_state, events);
        }

        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            let index = context.next_content_index;
            context.next_content_index += 1;
            let (mut new_state, mut events) = Self::start_text(index, context);
            if let Some(content) = &choice.delta.content
                && !content.is_empty()
            {
                let (text_state, text_events) =
                    Self::handle_text(choice, new_state.content_index().unwrap_or(index), context);
                new_state = text_state;
                events.extend(text_events);
            }
            return (new_state, events);
        }

        (ActiveState::Idle, vec![])
    }

    fn handle_thinking(
        choice: &OpenAIStreamChoice,
        content_index: u32,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        if let Some(tool_calls) = &choice.delta.tool_calls {
            let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                index: content_index,
            })];
            events.extend(handle_tool_calls_delta(tool_calls, context));
            return (ActiveState::Tool, events);
        }

        if let Some(finish_reason) = &choice.finish_reason {
            context.finish_reason = Some(finish_reason.clone());
            return (ActiveState::Thinking { content_index }, vec![]);
        }

        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
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
            return (new_state, events);
        }

        let mut events = vec![];
        if let Some(thinking) = &choice.delta.thinking {
            if let Some(content) = &thinking.content {
                events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::Thinking {
                        thinking: content.clone(),
                    },
                }));
            }
            if let Some(signature) = &thinking.signature {
                events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                    index: content_index,
                    delta: Delta::Signature {
                        signature: signature.clone(),
                    },
                }));
            }
        }

        if let Some(reasoning) = &choice.delta.reasoning {
            events.push(AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                index: content_index,
                delta: Delta::Thinking {
                    thinking: reasoning.clone(),
                },
            }));
        }

        (ActiveState::Thinking { content_index }, events)
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
        if let Some(tool_calls) = &choice.delta.tool_calls {
            let mut events = vec![AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                index: content_index,
            })];
            events.extend(handle_tool_calls_delta(tool_calls, context));
            return (ActiveState::Tool, events);
        }

        if let Some(finish_reason) = &choice.finish_reason {
            context.finish_reason = Some(finish_reason.clone());
            return (ActiveState::Text { content_index }, vec![]);
        }

        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            let event = AnthropicStreamEvent::ContentBlockDelta(ContentBlockDelta {
                index: content_index,
                delta: Delta::Text {
                    text: content.clone(),
                },
            });
            return (ActiveState::Text { content_index }, vec![event]);
        }

        (ActiveState::Text { content_index }, vec![])
    }

    fn handle_tool(
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        if let Some(finish_reason) = &choice.finish_reason {
            context.finish_reason = Some(finish_reason.clone());
            return (ActiveState::Tool, vec![]);
        }

        if choice.delta.content.is_some() {
            let mut events = Vec::new();
            if let Some(tool_index) = context.tool_index
                && let Some(tool_call) = context.tool_calls.get(&tool_index)
                && let Some(content_index) = tool_call.content_index
            {
                events.push(AnthropicStreamEvent::ContentBlockStop(ContentBlockStop {
                    index: content_index,
                }));
            }
            context.tool_index = None;

            let (new_state, new_events) = Self::start_text(context.next_content_index, context);
            events.extend(new_events);
            return (new_state, events);
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            let tool_events = handle_tool_calls_delta(tool_calls, context);
            return (ActiveState::Tool, tool_events);
        }

        (ActiveState::Tool, vec![])
    }
}
