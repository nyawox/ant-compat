use ant_compat::models::openai::{
    OpenAIDelta, OpenAIStreamChoice, OpenAIStreamChunk, OpenAIStreamFunction, OpenAIStreamToolCall,
    OpenAIUsage,
};

pub fn text_chunk(content: &str) -> OpenAIStreamChunk {
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                content: Some(content.to_string()),
                ..Default::default()
            },
            finish_reason: None,
        }],
        model: "kimi-k2-0711-preview".to_string(),
        usage: OpenAIUsage::default(),
    }
}

pub fn tool_chunk_partial(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    args: Option<&str>,
) -> OpenAIStreamChunk {
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                tool_calls: Some(vec![OpenAIStreamToolCall {
                    index,
                    id: id.map(|s| s.to_string()),
                    call_type: Some("function".to_string()),
                    function: Some(OpenAIStreamFunction {
                        name: name.map(|s| s.to_string()),
                        arguments: args.map(|s| s.to_string()),
                    }),
                }]),
                ..Default::default()
            },
            finish_reason: None,
        }],
        model: "bedrock/claude-sonnet-4-20250514".to_string(),
        usage: OpenAIUsage::default(),
    }
}

pub fn tool_chunk(index: u32, id: &str, name: &str, args: &str) -> OpenAIStreamChunk {
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                tool_calls: Some(vec![OpenAIStreamToolCall {
                    index,
                    id: Some(id.to_string()),
                    call_type: Some("function".to_string()),
                    function: Some(OpenAIStreamFunction {
                        name: Some(name.to_string()),
                        arguments: Some(args.to_string()),
                    }),
                }]),
                ..Default::default()
            },
            finish_reason: None,
        }],
        model: "kimi-k2-0711-preview".to_string(),
        usage: OpenAIUsage::default(),
    }
}

pub fn thinking_chunk(content: &str) -> OpenAIStreamChunk {
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                reasoning: Some(content.to_string()),
                ..Default::default()
            },
            finish_reason: None,
        }],
        model: "o3".to_string(),
        usage: OpenAIUsage::default(),
    }
}

pub fn final_chunk(reason: &str) -> OpenAIStreamChunk {
    OpenAIStreamChunk {
        id: "chatcmpl-123".to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta::default(),
            finish_reason: Some(reason.to_string()),
        }],
        model: "gpt-4.1-mini".to_string(),
        usage: OpenAIUsage::default(),
    }
}
