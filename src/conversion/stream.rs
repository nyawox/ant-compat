use async_stream::stream;
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::time;
use tracing::{debug, error};

use crate::models::{
    claude::FinishReason,
    claude::{
        AnthropicStreamEvent, ClaudeStreamMessage, ClaudeStreamUsage, ContentBlock,
        ContentBlockDelta, ContentBlockStart, ContentBlockStop, Delta, MessageDelta,
        MessageDeltaInfo, MessageStart, MessageStop,
    },
    openai::{OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamToolCall},
    shared::{ActiveState, MessageDeltaUsage, StreamState},
};

fn emit_event(event_type: &str, data: &impl serde::Serialize) -> Bytes {
    let data_str = serde_json::to_string(data).unwrap_or_default();
    debug!("Emitting event: {event_type}, data: {data_str}");
    Bytes::from(format!("event: {event_type}\ndata: {data_str}\n\n"))
}

fn emit_ping() -> Bytes {
    debug!("Emitting ping event");
    emit_event("ping", &json!({"type": "ping"}))
}

fn emit_initial_events(state: &StreamState) -> Vec<AnthropicStreamEvent> {
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

fn emit_final_events(
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

fn update_usage_from_chunk(chunk: &OpenAIStreamChunk, state: &mut StreamState) {
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

fn handle_tool_calls_delta(
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

pub fn convert_openai_stream_to_anthropic(
    response: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let message_id = format!("msg_{timestamp}");

    stream! {
        let mut stream_of_bytes = response.bytes_stream();
        let mut buffer = String::new();
        let mut state = StreamState {
            model: model.clone(),
            message_id: message_id.clone(),
            state: ActiveState::NotStarted,
            usage_data: MessageDeltaUsage::default(),
            next_content_index: 0,
            tool_calls: HashMap::new(),
            tool_index: None,
        };

        debug!("Starting stream conversion for model: {model}");

        for initial_event in emit_initial_events(&state) {
            let (event_type, data) = initial_event.to_parts();
            yield Ok(emit_event(event_type, &data));
        }

        yield Ok(emit_ping());

        let mut ping_interval = time::interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                chunk_result = stream_of_bytes.next() => {
                    let Some(chunk_result) = chunk_result else {
                        break;
                    };

                    if let ActiveState::Finished = state.state {
                        break;
                    }

                    let chunk_bytes = match chunk_result {
                        Ok(bytes) => {
                            debug!("Received chunk of size: {}", bytes.len());
                            bytes
                        },
                        Err(e) => {
                            let error_event = json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": e.to_string()
                                }
                            });
                            yield Ok(emit_event("error", &error_event));
                            yield Err(std::io::Error::other(e));
                            break;
                        }
                    };

                    buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));
                    let lines: Vec<&str> = buffer.split('\n').collect();
                    let new_buffer = (*lines.last().unwrap_or(&"")).to_string();

                    for line in &lines[..lines.len().saturating_sub(1)] {
                        if let ActiveState::Finished = state.state {
                            break;
                        }

                        if let Some(events) = process_stream_line(line, &mut state) {
                            for event in events {
                                let (event_type, data) = event.to_parts();
                                yield Ok(emit_event(event_type, &data));
                            }
                        }
                    }

                    buffer = new_buffer;
                }
                _ = ping_interval.tick() => {
                    if !matches!(state.state, ActiveState::Finished) {
                        yield Ok(emit_ping());
                    }
                }
            }

            if let ActiveState::Finished = state.state {
                break;
            }
        }
    }
}

fn process_stream_line(line: &str, state: &mut StreamState) -> Option<Vec<AnthropicStreamEvent>> {
    if !line.starts_with("data: ") {
        debug!(
            "Skipping non-data line: {}",
            line.chars().take(100).collect::<String>()
        );
        return None;
    }

    let data = &line[6..];
    if data == "[DONE]" {
        debug!("Stream finished with [DONE] marker");
        let choice = OpenAIStreamChoice {
            delta: OpenAIDelta::default(),
            finish_reason: Some("stop".to_string()),
            index: 0,
        };
        return Some(state.process_choice(&choice));
    }

    debug!(
        "Raw chunk data: {}",
        data.chars().take(500).collect::<String>()
    );

    let chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
        Ok(c) => {
            debug!("Successfully parsed OpenAI chunk: {:#?}", c);
            c
        }
        Err(e) => {
            error!(
                "Failed to parse stream chunk: {e}, data: {}",
                data.chars().take(200).collect::<String>()
            );
            return None;
        }
    };

    update_usage_from_chunk(&chunk, state);
    let mut all_events = Vec::new();
    for choice in chunk.choices {
        all_events.extend(state.process_choice(&choice));
    }

    Some(all_events)
}

impl StreamState {
    fn process_choice(&mut self, choice: &OpenAIStreamChoice) -> Vec<AnthropicStreamEvent> {
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
            ActiveState::NotStarted => Self::handle_not_started(choice, context),
            ActiveState::Thinking { content_index } => {
                Self::handle_thinking(choice, content_index, context)
            }
            ActiveState::Text { content_index } => {
                Self::handle_text(choice, content_index, context)
            }
            ActiveState::Tool => Self::handle_tool(choice, context),
            ActiveState::Finished => (ActiveState::Finished, vec![]),
        }
    }

    fn handle_not_started(
        choice: &OpenAIStreamChoice,
        context: &mut StreamState,
    ) -> (ActiveState, Vec<AnthropicStreamEvent>) {
        if let Some(tool_calls) = &choice.delta.tool_calls {
            let tool_events = handle_tool_calls_delta(tool_calls, context);
            return (ActiveState::Tool, tool_events);
        }

        if let Some(finish_reason) = &choice.finish_reason {
            let final_events = emit_final_events(ActiveState::NotStarted, context, finish_reason);
            return (ActiveState::Finished, final_events);
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

        (ActiveState::NotStarted, vec![])
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
            let final_events = emit_final_events(
                ActiveState::Thinking { content_index },
                context,
                finish_reason,
            );
            return (ActiveState::Finished, final_events);
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
            let final_events =
                emit_final_events(ActiveState::Text { content_index }, context, finish_reason);
            return (ActiveState::Finished, final_events);
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
            let final_events = emit_final_events(ActiveState::Tool, context, finish_reason);
            return (ActiveState::Finished, final_events);
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
