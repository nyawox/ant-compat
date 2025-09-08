use std::sync::LazyLock;

use regex::Regex;
use tracing::debug;

use crate::{
    lazy_regex,
    models::claude::{
        ClaudeContent, ClaudeContentBlock, ClaudeMessagesRequest, ClaudeSystem, ClaudeThinking,
    },
    utils::map_reasoning_effort_to_budget_tokens,
};

use super::models::{Condition, ProxyDirective, ResponsesSettings, Settings};

static DIRECTIVE_REGEX: LazyLock<Regex> =
    lazy_regex!(r"(?s)---\s*PROXY DIRECTIVE\s*---\s*(.*?)\s*---\s*END DIRECTIVE\s*---");

const CLAUDE_MD_MARKER: &str = "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# claudeMd";

#[derive(Debug, Default)]
pub struct DirectiveProcessor;

impl DirectiveProcessor {
    pub fn process(request: &mut ClaudeMessagesRequest) -> Settings {
        let directive = Self::find_directive(request);

        if let Some(dir) = directive {
            let settings = Self::resolve_settings(request, &dir);
            Self::apply_parameters(request, &settings);
            settings
        } else {
            Settings::default()
        }
    }

    fn find_directive(request: &mut ClaudeMessagesRequest) -> Option<ProxyDirective> {
        if let Some(system_prompt) = &mut request.system
            && let Some(directive) = Self::extract_from_system(system_prompt)
        {
            debug!("Directive extracted from subagent system prompt.");
            return Some(directive);
        }
        // CLAUDE.md is usually in the first user message,
        // but after context summarization it may appear after Read tool outputs in the request array
        let user_directive = request
            .messages
            .iter_mut()
            .filter(|message| message.role == "user")
            .enumerate()
            .find_map(|(user_index, message)| {
                let begins_with_marker = match &message.content {
                    ClaudeContent::Text(text) => text.starts_with(CLAUDE_MD_MARKER),
                    ClaudeContent::Array(blocks) => blocks.iter().any(|block| {
                        block
                            .text
                            .as_ref()
                            .is_some_and(|text| text.starts_with(CLAUDE_MD_MARKER))
                    }),
                };
                // Enabling LIMIT_DIRECTIVE_TO_CLAUDEMD will break directives in zed rules (or other clients)
                // for security, non-first user messages require the claudemd marker at the start
                let limit_enabled = std::env::var("LIMIT_DIRECTIVE_TO_CLAUDEMD").is_ok();
                let should_extract = if user_index == 0 {
                    !limit_enabled || begins_with_marker
                } else {
                    begins_with_marker
                };
                if should_extract {
                    Self::extract_from_content(&mut message.content)
                } else {
                    None
                }
            });

        if let Some(directive) = user_directive {
            debug!("Directive extracted from user message.");
            return Some(directive);
        }
        None
    }

    fn resolve_settings(request: &ClaudeMessagesRequest, directive: &ProxyDirective) -> Settings {
        let base = directive.global.clone().unwrap_or_default();
        directive
            .rules
            .iter()
            .filter(|rule| Self::evaluate_condition(request, &rule.if_clause))
            .fold(base, |accumulated, rule| {
                Self::merge_settings(accumulated, &rule.apply)
            })
    }

    fn extract_from_system(system_prompt_enum: &mut ClaudeSystem) -> Option<ProxyDirective> {
        let mut content = match system_prompt_enum {
            ClaudeSystem::Text(text) => ClaudeContent::Text(text.clone()),
            ClaudeSystem::Array(blocks) => ClaudeContent::Array(blocks.clone()),
        };

        let directive = Self::extract_from_content(&mut content);

        if directive.is_some() {
            match content {
                ClaudeContent::Text(text) => *system_prompt_enum = ClaudeSystem::Text(text),
                ClaudeContent::Array(blocks) => *system_prompt_enum = ClaudeSystem::Array(blocks),
            }
        }
        directive
    }

    fn extract_from_content(content: &mut ClaudeContent) -> Option<ProxyDirective> {
        match content {
            ClaudeContent::Text(text) => {
                let (cleaned_text, directive) = Self::parse_directive_from_text(text);
                if directive.is_some() {
                    *text = cleaned_text;
                }
                directive
            }
            ClaudeContent::Array(blocks) => {
                let mut directive = None;
                let mut found_in_block = false;

                for block in blocks.iter_mut() {
                    if found_in_block {
                        continue;
                    }
                    if let ClaudeContentBlock {
                        block_type,
                        text: Some(text),
                        ..
                    } = block
                        && block_type == "text"
                        && DIRECTIVE_REGEX.is_match(text)
                    {
                        let (cleaned_text, parsed_directive) =
                            Self::parse_directive_from_text(text);
                        if parsed_directive.is_some() {
                            *text = cleaned_text;
                            directive = parsed_directive;
                            found_in_block = true;
                        }
                    }
                }
                directive
            }
        }
    }

    fn parse_directive_from_text(text: &str) -> (String, Option<ProxyDirective>) {
        if let Some(captures) = DIRECTIVE_REGEX.captures(text) {
            let directive_json = captures.get(1).map_or("", |m| m.as_str()).trim();
            debug!("Extracted directive JSON: {directive_json}");
            let directive = match serde_json::from_str(directive_json) {
                Ok(dir) => Some(dir),
                Err(e) => {
                    debug!("Failed to parse directive JSON: {e}");
                    None
                }
            };
            let cleaned_text = DIRECTIVE_REGEX.replace(text, "").to_string();
            (cleaned_text, directive)
        } else {
            (text.to_string(), None)
        }
    }

    fn evaluate_condition(request: &ClaudeMessagesRequest, condition: &Condition) -> bool {
        match condition {
            Condition::ModelContains(substring) => request.model.contains(substring),
        }
    }

    fn apply_parameters(request: &mut ClaudeMessagesRequest, settings: &Settings) {
        if let Some(model) = &settings.model {
            request.model.clone_from(model);
        }
        if let Some(max_tokens) = settings.max_tokens {
            request.max_tokens = max_tokens;
        }
        if let Some(temperature) = settings.temperature {
            request.temperature = Some(temperature);
        }
        if let Some(top_p) = settings.top_p {
            request.top_p = Some(top_p);
        }
        if let Some(reasoning_effort) = &settings.reasoning_effort {
            let budget_tokens = map_reasoning_effort_to_budget_tokens(reasoning_effort.as_str());
            request.thinking = Some(ClaudeThinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(budget_tokens),
            });
        }
    }

    fn merge_settings(mut accumulated: Settings, incoming: &Settings) -> Settings {
        if incoming.model.is_some() {
            accumulated.model.clone_from(&incoming.model);
        }
        if incoming.max_tokens.is_some() {
            accumulated.max_tokens = incoming.max_tokens;
        }
        if incoming.temperature.is_some() {
            accumulated.temperature = incoming.temperature;
        }
        if incoming.top_p.is_some() {
            accumulated.top_p = incoming.top_p;
        }
        if incoming.reasoning_effort.is_some() {
            accumulated
                .reasoning_effort
                .clone_from(&incoming.reasoning_effort);
        }
        if incoming.enable_meowsings.is_some() {
            accumulated.enable_meowsings = incoming.enable_meowsings;
        }
        if incoming.responses.is_some() {
            accumulated.responses =
                Self::merge_responses(accumulated.responses, incoming.responses.clone());
        }
        accumulated
    }

    fn merge_responses(
        base: Option<ResponsesSettings>,
        incoming: Option<ResponsesSettings>,
    ) -> Option<ResponsesSettings> {
        match (base, incoming) {
            (None, None) => None,
            (Some(base), None) => Some(base),
            (None, Some(incoming)) => Some(incoming),
            (Some(mut base), Some(incoming)) => {
                if incoming.enable.is_some() {
                    base.enable = incoming.enable;
                }
                if incoming.max_output_tokens.is_some() {
                    base.max_output_tokens = incoming.max_output_tokens;
                }
                if incoming.reasoning_summary.is_some() {
                    base.reasoning_summary = incoming.reasoning_summary;
                }
                Some(base)
            }
        }
    }
}
