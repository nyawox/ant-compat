use serde_json::{Map, Value, json};

#[must_use]
pub fn convert_openai_to_claude(openai_response: &Value, model: &str) -> Value {
    let choice = &openai_response["choices"][0];
    let message = &choice["message"];
    let mut content_blocks = Vec::new();

    // I could not figure out how to get reasoning working for non-streaming
    // PR welcome :)

    // if let Some(reasoning) = message["reasoning"].as_str().filter(|s| !s.is_empty()) {
    //     content_blocks.push(json!({
    //         "type": "thinking",
    //         "thinking": reasoning,
    //     }));
    // }

    // if let Some(content) = message["content"].as_str().filter(|s| !s.is_empty()) {
    //     content_blocks.push(json!({
    //         "type": "text",
    //         "text": content
    //     }));
    // }

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
