use std::{collections::HashMap, pin::Pin};

use async_stream::stream;
use futures_util::{Stream, StreamExt, TryStreamExt};
use serde_json::{Map, Value, json};
use tokio_sse_codec::{Frame, SseDecoder};
use tokio_util::codec::FramedRead;
use tokio_util::io::StreamReader;
use tracing::debug;

use crate::{
    adapters::traits::ApiAdapter,
    conversion::request::Request,
    error::AppError,
    models::openai::{
        OpenAIContent, OpenAIDelta, OpenAIRequest, OpenAIStreamChoice, OpenAIStreamChunk,
        OpenAIStreamFunction, OpenAIStreamToolCall, OpenAIUsage,
    },
};

pub struct ResponsesApiAdapter {
    pub max_output_tokens: Option<u32>,
    pub reasoning_summary: Option<String>,
}

impl ResponsesApiAdapter {
    fn map_messages_to_input(messages: &[crate::models::openai::OpenAIMessage]) -> Value {
        let mut items: Vec<Value> = Vec::new();
        for message in messages {
            let role = message.role.as_str();
            if role == "system" {
                continue;
            }
            if role == "tool" {
                items.push(Self::map_tool_output_item(message));
                continue;
            }
            items.push(Self::map_message_item(role, message.content.as_ref()));
            items.extend(Self::map_function_call_items(message.tool_calls.as_ref()));
        }
        Value::Array(items)
    }

    fn map_tool_output_item(message: &crate::models::openai::OpenAIMessage) -> Value {
        let call_id = message.tool_call_id.clone().unwrap_or_default();
        let output = Self::extract_tool_output_string(message.content.as_ref());
        json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        })
    }

    fn extract_tool_output_string(content: Option<&OpenAIContent>) -> String {
        match content {
            Some(OpenAIContent::Text(text)) => text.clone(),
            Some(OpenAIContent::Array(parts)) => {
                let mut text_buffer = String::new();
                for part in parts {
                    if let Some(text) = &part.text {
                        if !text_buffer.is_empty() {
                            text_buffer.push('\n');
                        }
                        text_buffer.push_str(text);
                    }
                }
                if text_buffer.is_empty() {
                    serde_json::to_string(parts).unwrap_or_default()
                } else {
                    text_buffer
                }
            }
            None => String::new(),
        }
    }

    fn map_message_item(role: &str, content: Option<&OpenAIContent>) -> Value {
        match content {
            Some(OpenAIContent::Text(text)) => {
                let part_type = Self::part_type_for_role(role);
                json!({
                    "type": "message",
                    "role": role,
                    "content": [{"type": part_type, "text": text}],
                })
            }
            Some(OpenAIContent::Array(parts)) => {
                let content_parts = Self::map_array_parts_for_role(role, parts);
                json!({
                    "type": "message",
                    "role": role,
                    "content": content_parts,
                })
            }
            None => {
                json!({
                    "type": "message",
                    "role": role,
                    "content": [],
                })
            }
        }
    }

    fn part_type_for_role(role: &str) -> &'static str {
        if role == "assistant" {
            "output_text"
        } else {
            "input_text"
        }
    }

    fn map_array_parts_for_role(
        role: &str,
        parts: &[crate::models::openai::OpenAIContentPart],
    ) -> Vec<Value> {
        parts
            .iter()
            .filter_map(|part| {
                if part.part_type == "text" {
                    let part_type = Self::part_type_for_role(role);
                    part.text
                        .as_ref()
                        .map(|text| json!({"type": part_type, "text": text}))
                } else if part.part_type == "image_url" {
                    if role == "assistant" {
                        None
                    } else {
                        part.image_url.as_ref().map(|url| {
                            json!({
                                "type": "input_image",
                                "image_url": url.url
                            })
                        })
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    fn map_function_call_items(
        tool_calls: Option<&Vec<crate::models::openai::OpenAIToolCall>>,
    ) -> Vec<Value> {
        tool_calls
            .map(|calls| {
                calls
                    .iter()
                    .filter(|call| call.call_type == "function")
                    .map(|call| {
                        json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        })
                    })
                    .collect::<Vec<Value>>()
            })
            .unwrap_or_default()
    }

    fn handle_output_item_added(
        parsed: &Value,
        call_index_map: &mut HashMap<String, u32>,
        next_index: &mut u32,
        model: &str,
    ) -> Option<OpenAIStreamChunk> {
        let item = parsed.get("item")?;
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        if item_type != "function_call" {
            return None;
        }
        let call_id_opt = item.get("call_id").and_then(Value::as_str);
        let item_id_opt = item.get("id").and_then(Value::as_str);
        let existing_index = call_id_opt
            .and_then(|id| call_index_map.get(id).copied())
            .or_else(|| item_id_opt.and_then(|id| call_index_map.get(id).copied()));
        let index = if let Some(i) = existing_index {
            i
        } else {
            let i = *next_index;
            *next_index += 1;
            if let Some(id) = call_id_opt {
                call_index_map.insert(id.to_string(), i);
            }
            if let Some(id) = item_id_opt {
                call_index_map.insert(id.to_string(), i);
            }
            i
        };
        let name = item.get("name").and_then(Value::as_str).map(str::to_string);
        let call_id = call_id_opt.or(item_id_opt).unwrap_or("").to_string();
        let tool_call = OpenAIStreamToolCall {
            index,
            id: Some(call_id),
            call_type: Some("function".to_string()),
            function: Some(OpenAIStreamFunction {
                name,
                arguments: None,
            }),
        };
        let chunk = OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    tool_calls: Some(vec![tool_call]),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: model.to_string(),
            usage: OpenAIUsage::default(),
        };
        Some(chunk)
    }

    fn handle_reasoning_summary_text_delta(
        parsed: &Value,
        model: &str,
    ) -> Option<OpenAIStreamChunk> {
        let delta = parsed.get("delta").and_then(Value::as_str)?;
        let chunk = OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    reasoning_content: Some(delta.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: model.to_string(),
            usage: OpenAIUsage::default(),
        };
        Some(chunk)
    }

    fn handle_output_text_delta(parsed: &Value, model: &str) -> Option<OpenAIStreamChunk> {
        let delta = parsed.get("delta").and_then(Value::as_str)?;
        let chunk = OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    content: Some(delta.to_string()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: model.to_string(),
            usage: OpenAIUsage::default(),
        };
        Some(chunk)
    }

    fn handle_function_call_arguments_delta(
        parsed: &Value,
        call_index_map: &mut HashMap<String, u32>,
        next_index: &mut u32,
        model: &str,
    ) -> Option<OpenAIStreamChunk> {
        let call_key = parsed
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| parsed.get("item_id").and_then(Value::as_str))
            .or_else(|| parsed.get("id").and_then(Value::as_str))
            .unwrap_or("");
        if call_key.is_empty() {
            return None;
        }
        let index = if let Some(i) = call_index_map.get(call_key) {
            *i
        } else {
            let i = *next_index;
            *next_index += 1;
            call_index_map.insert(call_key.to_string(), i);
            i
        };
        let delta = parsed.get("delta").and_then(Value::as_str)?.to_string();
        debug!("Responses function_call_arguments.delta call_id={call_key} delta={delta}");
        let tool_call = OpenAIStreamToolCall {
            index,
            id: None,
            call_type: Some("function".to_string()),
            function: Some(OpenAIStreamFunction {
                name: None,
                arguments: Some(delta),
            }),
        };
        let chunk = OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    tool_calls: Some(vec![tool_call]),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            model: model.to_string(),
            usage: OpenAIUsage::default(),
        };
        Some(chunk)
    }

    fn handle_output_item_done(parsed: &Value, model: &str) -> Option<OpenAIStreamChunk> {
        let item = parsed.get("item")?;
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        if item_type != "function_call" {
            return None;
        }
        let chunk = OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta::default(),
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: model.to_string(),
            usage: OpenAIUsage::default(),
        };
        Some(chunk)
    }

    fn handle_completed(parsed: &Value, model: &str) -> OpenAIStreamChunk {
        let usage = parsed
            .get("response")
            .and_then(|r| r.get("usage"))
            .cloned()
            .unwrap_or(json!({}));
        let prompt_tokens = u32::try_from(
            usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);
        let completion_tokens = u32::try_from(
            usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);
        OpenAIStreamChunk {
            id: "resp.stream".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta::default(),
                finish_reason: Some("stop".to_string()),
            }],
            model: model.to_string(),
            usage: OpenAIUsage {
                prompt_tokens,
                completion_tokens,
                prompt_tokens_details: None,
            },
        }
    }
}

impl ApiAdapter for ResponsesApiAdapter {
    fn endpoint_suffix(&self) -> &'static str {
        "/responses"
    }

    fn build_body(&self, openai_req: &OpenAIRequest, _original: &Request) -> Value {
        let instructions = openai_req
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .filter_map(|m| match &m.content {
                Some(OpenAIContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let input = Self::map_messages_to_input(&openai_req.messages);

        let mut body = Map::new();
        let model = openai_req
            .model
            .split_once('.')
            .map_or_else(|| openai_req.model.clone(), |(_, m)| m.to_string());
        body.insert("model".to_string(), Value::String(model));
        body.insert("input".to_string(), input);
        body.insert(
            "stream".to_string(),
            Value::Bool(openai_req.stream.unwrap_or(false)),
        );
        body.insert("store".to_string(), Value::Bool(false));
        if !instructions.is_empty() {
            body.insert("instructions".to_string(), Value::String(instructions));
        }
        let summary = self
            .reasoning_summary
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        if let Some(effort) = &openai_req.reasoning_effort {
            body.insert(
                "reasoning".to_string(),
                json!({"effort": effort, "summary": summary}),
            );
        } else {
            body.insert("reasoning".to_string(), json!({"summary": summary}));
        }
        if let Some(t) = openai_req.temperature {
            body.insert("temperature".to_string(), json!(t));
        }
        if let Some(p) = openai_req.top_p {
            body.insert("top_p".to_string(), json!(p));
        }
        if let Some(max_out) = self.max_output_tokens {
            body.insert("max_output_tokens".to_string(), json!(max_out));
        }
        if let Some(tools) = &openai_req.tools {
            let mapped_tools: Vec<Value> = tools
                .iter()
                .filter_map(|t| {
                    if t.tool_type == "function" {
                        Some(json!({
                            "type": "function",
                            "name": t.function.name,
                            "parameters": t.function.parameters,
                            "strict": false,
                            "description": t.function.description,
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            if !mapped_tools.is_empty() {
                body.insert("tools".to_string(), Value::Array(mapped_tools));
            }
        }
        if let Some(choice) = &openai_req.tool_choice {
            let mapped_choice = match choice {
                crate::models::openai::OpenAIToolChoice::String(mode) => json!(mode),
                crate::models::openai::OpenAIToolChoice::Object {
                    choice_type,
                    function,
                } => {
                    if choice_type == "function" {
                        json!({"type": "function", "name": function.name})
                    } else {
                        json!("auto")
                    }
                }
            };
            body.insert("tool_choice".to_string(), mapped_choice);
        }
        Value::Object(body)
    }

    fn normalize_non_stream_json(&self, upstream_json: Value) -> Value {
        let output = upstream_json
            .get("output")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut text_acc = String::new();
        for item in &output {
            if item.get("type").and_then(|v| v.as_str()) == Some("message")
                && let Some(content) = item.get("content").and_then(|v| v.as_array())
            {
                for c in content {
                    if c.get("type").and_then(|v| v.as_str()) == Some("output_text")
                        && let Some(t) = c.get("text").and_then(|v| v.as_str())
                    {
                        if !text_acc.is_empty() {
                            text_acc.push('\n');
                        }
                        text_acc.push_str(t);
                    }
                }
            }
        }
        let mut tool_calls = Vec::new();
        for item in &output {
            if item.get("type").and_then(|v| v.as_str()) == Some("function_call")
                && let (Some(name), Some(call_id)) = (
                    item.get("name").and_then(|v| v.as_str()),
                    item.get("call_id").and_then(|v| v.as_str()),
                )
            {
                let arguments = item
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap_or("").to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();
                tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }));
            }
        }
        let status = upstream_json
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let finish_reason = if status == "incomplete" {
            "length"
        } else {
            "stop"
        };
        let usage = upstream_json
            .get("usage")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let prompt_tokens = u32::try_from(
            usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);
        let completion_tokens = u32::try_from(
            usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);
        json!({
            "id": upstream_json.get("id").cloned().unwrap_or(json!("resp")),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": text_acc,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        })
    }

    fn chunk_stream(
        &self,
        response: reqwest::Response,
        original: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, AppError>> + Send>> {
        let model = original.model.clone();
        let mut call_index_map: HashMap<String, u32> = HashMap::new();
        let mut next_index: u32 = 0;
        let byte_stream = response.bytes_stream().map_err(AppError::from);
        let stream_reader = StreamReader::new(byte_stream);
        let sse_stream = FramedRead::new(stream_reader, SseDecoder::<String>::new());
        Box::pin(stream! {
            futures_util::pin_mut!(sse_stream);
            while let Some(frame) = sse_stream.next().await {
                match frame {
                    Ok(Frame::Event(event)) => {
                        let data = &event.data;
                        if data == "[DONE]" { continue; }
                        let parsed: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => { yield Err(AppError::StreamError(format!("Failed to parse responses event: {e}, data: {data}"))); continue; }
                        };
                        let event_type = parsed.get("type").and_then(Value::as_str).unwrap_or("");
                        // suppress noisy events
                        if !matches!(event_type, "response.in_progress" | "response.created" | "response.completed") {
                            debug!("Responses event type: {event_type}");
                            debug!("Responses raw event: {data}");
                        }
                        let handled_chunk = match event_type {
                            "response.output_item.added" => Self::handle_output_item_added(&parsed, &mut call_index_map, &mut next_index, &model),
                            "response.reasoning_summary_text.delta" => Self::handle_reasoning_summary_text_delta(&parsed, &model),
                            "response.output_text.delta" => Self::handle_output_text_delta(&parsed, &model),
                            "response.function_call_arguments.delta" => Self::handle_function_call_arguments_delta(&parsed, &mut call_index_map, &mut next_index, &model),
                            "response.output_item.done" => Self::handle_output_item_done(&parsed, &model),
                            "response.completed" => {
                                let chunk = Self::handle_completed(&parsed, &model);
                                yield Ok(chunk);
                                break;
                            }
                            _ => None,
                        };
                        if let Some(chunk) = handled_chunk { yield Ok(chunk); }
                    }
                    Ok(_) => {}
                    Err(e) => { debug!("Upstream SSE error: {e}"); break; }
                }
            }
        })
    }
}
