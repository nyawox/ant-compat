use crate::adapters::RequestAdapter;
use tracing::debug;
pub type Request = ClaudeMessagesRequest;
use crate::{
    models::{
        claude::{
            ClaudeContent, ClaudeContentBlock, ClaudeMessage, ClaudeMessagesRequest, ClaudeSystem,
            ClaudeTool, ClaudeToolChoice,
        },
        openai::{
            OpenAIContent, OpenAIContentPart, OpenAIFunction, OpenAIFunctionChoice, OpenAIImageUrl,
            OpenAIMessage, OpenAIRequest, OpenAITool, OpenAIToolCall, OpenAIToolChoice,
            OpenAIToolFunction, StreamOptions,
        },
    },
    utils::map_budget_tokens_to_reasoning_effort,
};

#[must_use]
pub fn convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest,
    model_name: &str,
    adapter: &RequestAdapter,
) -> OpenAIRequest {
    let req_clone = claude_request.clone();
    let mut messages = Vec::new();

    if let Some(system) = &claude_request.system {
        let mut system_content = match system {
            ClaudeSystem::Text(text) => text.clone(),
            ClaudeSystem::Array(blocks) => blocks
                .iter()
                .filter_map(|block| {
                    if block.block_type == "text" {
                        block.text.as_ref()
                    } else {
                        None
                    }
                })
                .cloned()
                .collect::<Vec<String>>()
                .join("\n"),
        };

        system_content = adapter.adapt_system_prompt(&system_content, &req_clone);
        if !system_content.is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::Text(system_content)),
                ..Default::default()
            });
        }
    }

    for message in &claude_request.messages {
        convert_claude_message_to_openai(message, &mut messages, adapter, &req_clone);
    }

    let reasoning_effort = claude_request
        .thinking
        .as_ref()
        .filter(|t| t.thinking_type == "enabled")
        .and_then(|t| t.budget_tokens.map(map_budget_tokens_to_reasoning_effort));

    let mut openai_request = OpenAIRequest {
        model: adapter.adapt_model(model_name, &req_clone),
        messages,
        max_tokens: None,
        max_completion_tokens: None,
        temperature: claude_request.temperature,
        top_p: claude_request.top_p,
        stream: claude_request.stream,
        stop: claude_request.stop_sequences,
        tools: None,
        tool_choice: None,
        reasoning_effort,
        stream_options: if claude_request.stream.unwrap_or(false) {
            Some(StreamOptions {
                include_usage: Some(true),
            })
        } else {
            None
        },
    };

    let adapted_tools = adapter.adapt_tools(claude_request.tools, &req_clone);
    if let Some(tools) = adapted_tools
        && !tools.is_empty()
    {
        openai_request.tools = Some(convert_claude_tools_to_openai(tools, adapter, &req_clone));
    }

    let adapted_tool_choice = adapter.adapt_tool_choice(claude_request.tool_choice, &req_clone);
    if let Some(tool_choice) = adapted_tool_choice {
        openai_request.tool_choice = Some(convert_claude_tool_choice_to_openai(tool_choice));
    }

    openai_request.temperature = adapter.adapt_temperature(openai_request.temperature, &req_clone);
    openai_request.top_p = adapter.adapt_top_p(openai_request.top_p, &req_clone);
    openai_request.max_tokens = adapter.adapt_max_tokens(claude_request.max_tokens, &req_clone);
    openai_request.max_completion_tokens =
        adapter.adapt_max_completion_tokens(claude_request.max_tokens, &req_clone);

    openai_request.messages = adapter.adapt_messages(openai_request.messages, &req_clone);

    openai_request
}

fn convert_claude_message_to_openai(
    message: &ClaudeMessage,
    messages: &mut Vec<OpenAIMessage>,
    adapter: &RequestAdapter,
    request: &Request,
) {
    let mut adapted_message = message.clone();
    if let ("user", ClaudeContent::Text(text)) =
        (adapted_message.role.as_str(), &adapted_message.content)
    {
        adapted_message.content = ClaudeContent::Text(adapter.adapt_user_prompt(text, request));
    }
    match adapted_message.role.as_str() {
        "user" => convert_claude_user_message(adapted_message.content, messages, adapter, request),
        "assistant" => convert_claude_assistant_message(adapted_message.content, messages),
        _ => {}
    }
}

fn convert_claude_user_message(
    content: ClaudeContent,
    messages: &mut Vec<OpenAIMessage>,
    adapter: &RequestAdapter,
    request: &Request,
) {
    match content {
        ClaudeContent::Text(text) => {
            messages.push(OpenAIMessage {
                role: "user".to_string(),
                content: Some(OpenAIContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        ClaudeContent::Array(blocks) => {
            convert_claude_content_blocks(&blocks, messages, adapter, request);
        }
    }
}

fn convert_claude_content_blocks(
    blocks: &[ClaudeContentBlock],
    messages: &mut Vec<OpenAIMessage>,
    adapter: &RequestAdapter,
    request: &Request,
) {
    let tool_results: Vec<_> = blocks
        .iter()
        .filter(|b| b.block_type == "tool_result")
        .collect();

    for block in &tool_results {
        if let (Some(tool_use_id), Some(content)) = (&block.tool_use_id, &block.content) {
            let tool_name = request
                .find_tool_name_by_id(tool_use_id)
                .unwrap_or_default();
            let final_content = if let serde_json::Value::String(s) = content {
                adapter.adapt_tool_result(&tool_name, s, request)
            } else {
                serde_json::to_string(content).unwrap_or_default()
            };
            debug!(
                "
--- ADAPTED TOOL RESULT ---
{final_content}
--- END ADAPTED TOOL RESULT ---"
            );
            messages.push(OpenAIMessage {
                role: "tool".to_string(),
                content: Some(OpenAIContent::Text(final_content)),
                tool_calls: None,
                tool_call_id: Some(tool_use_id.clone()),
            });
        }
    }

    let other_content: Vec<_> = blocks
        .iter()
        .filter(|b| b.block_type != "tool_result")
        .collect();

    if !other_content.is_empty() {
        let content_parts: Vec<OpenAIContentPart> = other_content
            .iter()
            .filter_map(|block| match block.block_type.as_str() {
                "text" => block.text.as_ref().map(|text| OpenAIContentPart {
                    part_type: "text".to_string(),
                    text: Some(adapter.adapt_user_prompt(text, request)),
                    image_url: None,
                }),
                "image" => block.source.as_ref().map(|source| OpenAIContentPart {
                    part_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(OpenAIImageUrl {
                        url: format!("data:{};base64,{}", source.media_type, source.data),
                    }),
                }),
                _ => None,
            })
            .collect();

        if !content_parts.is_empty() {
            messages.push(OpenAIMessage {
                role: "user".to_string(),
                content: Some(OpenAIContent::Array(content_parts)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }
}

fn convert_claude_assistant_message(content: ClaudeContent, messages: &mut Vec<OpenAIMessage>) {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    match content {
        ClaudeContent::Text(text) => {
            text_parts.push(text);
        }
        ClaudeContent::Array(blocks) => {
            for block in blocks {
                match block.block_type.as_str() {
                    "text" => {
                        if let Some(text) = block.text {
                            text_parts.push(text);
                        }
                    }
                    "tool_use" => {
                        if let (Some(id), Some(name), Some(input)) =
                            (block.id, block.name, block.input)
                        {
                            tool_calls.push(OpenAIToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: OpenAIFunction {
                                    name,
                                    arguments: serde_json::to_string(&input).unwrap_or_default(),
                                },
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // WORKAROUND: Unable to submit request because it must include at least one parts field, which describes the prompt input. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini
    if text_parts.is_empty() && tool_calls.is_empty() {
        return;
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(OpenAIContent::Text(text_parts.join("\n")))
    };

    messages.push(OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
    });
}

fn convert_claude_tools_to_openai(
    tools: Vec<ClaudeTool>,
    adapter: &RequestAdapter,
    request: &Request,
) -> Vec<OpenAITool> {
    tools
        .into_iter()
        .map(|tool| {
            let parameters = adapter.adapt_tool_schema(&tool.input_schema, request);
            let description = tool
                .description
                .as_ref()
                .map(|d| adapter.adapt_tool_description(d, request))
                .unwrap_or_default();
            OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIToolFunction {
                    name: tool.name,
                    description: Some(description),
                    parameters,
                },
            }
        })
        .collect()
}

fn convert_claude_tool_choice_to_openai(tool_choice: ClaudeToolChoice) -> OpenAIToolChoice {
    match tool_choice.choice_type.as_str() {
        "tool" => {
            if let Some(name) = tool_choice.name {
                OpenAIToolChoice::Object {
                    choice_type: "function".to_string(),
                    function: OpenAIFunctionChoice { name },
                }
            } else {
                OpenAIToolChoice::String("auto".to_string())
            }
        }
        _ => OpenAIToolChoice::String("auto".to_string()),
    }
}
