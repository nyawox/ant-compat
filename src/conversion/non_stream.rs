use super::think_parser::ThinkTagParser;
use serde_json::{Map, Value, json};

fn parse_text_blocks(input: &str) -> Vec<Value> {
    let mut parser = ThinkTagParser::new();
    if input.is_empty() {
        return Vec::new();
    }
    let mut remaining = parser.preprocess(input);
    let mut blocks = Vec::new();
    while !remaining.is_empty() {
        if matches!(
            parser.state,
            super::think_parser::ThinkParserState::Disabled
        ) {
            blocks.push(json!({"type": "text", "text": remaining}));
            break;
        }
        if let Some(pos) = remaining
            .find("<think>")
            .or_else(|| remaining.find("<cot>"))
        {
            let before = &remaining[..pos];
            if !before.is_empty() {
                blocks.push(json!({"type": "text", "text": before}));
            }
            let after_start = parser.clean_before(&remaining[pos..]);
            if let Some(end_pos) = after_start
                .find("</think>")
                .or_else(|| after_start.find("</cot>"))
                .or_else(|| after_start.find("<end_cot>"))
            {
                let thinking = &after_start[..end_pos];
                if !thinking.is_empty() {
                    blocks.push(json!({"type": "thinking", "thinking": thinking}));
                }
                let tail = &after_start[end_pos..];
                remaining = parser.clean_after(tail);
                parser.on_think_end();
            } else {
                if !after_start.is_empty() {
                    blocks.push(json!({"type": "thinking", "thinking": after_start}));
                }
                break;
            }
        } else {
            blocks.push(json!({"type": "text", "text": remaining}));
            break;
        }
    }
    blocks
}

#[must_use]
pub fn convert_openai_to_claude(openai_response: &Value, model: &str) -> Value {
    let choice = &openai_response["choices"][0];
    let message = &choice["message"];
    let mut content_blocks = Vec::new();

    if let Some(reasoning) = message["reasoning_content"]
        .as_str()
        .filter(|s| !s.is_empty())
    {
        content_blocks.push(json!({
            "type": "thinking",
            "thinking": reasoning,
        }));
    }

    if let Some(content) = message["content"].as_str().filter(|s| !s.is_empty()) {
        content_blocks.extend(parse_text_blocks(content));
    }

    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tool_call in tool_calls {
            if let (Some(id), Some(name), Some(arguments)) = (
                tool_call["id"].as_str(),
                tool_call["function"]["name"].as_str(),
                tool_call["function"]["arguments"].as_str(),
            ) {
                let input: Value =
                    serde_json::from_str(arguments).unwrap_or(Value::Object(Map::new()));
                content_blocks.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
            }
        }
    }

    let stop_reason = match choice["finish_reason"].as_str() {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        Some("content_filter") => "stop_sequence",
        _ => "end_turn",
    };

    json!({
        "id": openai_response["id"],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": openai_response["usage"]["prompt_tokens"],
            "output_tokens": openai_response["usage"]["completion_tokens"]
        }
    })
}
