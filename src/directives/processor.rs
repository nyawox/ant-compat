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

use super::models::{Condition, ProxyDirective, Settings};

static DIRECTIVE_REGEX: LazyLock<Regex> =
    lazy_regex!(r"(?s)---\s*PROXY DIRECTIVE\s*---\s*(.*?)\s*---\s*END DIRECTIVE\s*---");

// we currently limit directive processing to the very first message with `user` role (which contains CLAUDE.md).
// while this is acceptable, at least for now
// it raises security questions about how far this directive capability should be extended.
// should consider what limitations to impose.
const CLAUDE_MD_MARKER: &str = "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# claudeMd";

#[derive(Debug, Default)]
pub struct DirectiveProcessor;

impl DirectiveProcessor {
    pub fn process(request: &mut ClaudeMessagesRequest) {
        if let Some(system_prompt_enum) = &mut request.system
            && let Some(dir) = Self::extract_from_system(system_prompt_enum)
        {
            debug!("Directive applied from subagent system prompt.");
            Self::apply_directive(request, &dir);
            return;
        }

        if let Some(first_user_message) = request.messages.iter_mut().find(|m| m.role == "user") {
            let has_marker = match &first_user_message.content {
                ClaudeContent::Text(text) => text.contains(CLAUDE_MD_MARKER),
                ClaudeContent::Array(blocks) => blocks.iter().any(|b| {
                    b.text
                        .as_ref()
                        .is_some_and(|t| t.contains(CLAUDE_MD_MARKER))
                }),
            };

            if has_marker
                && let Some(dir) = Self::extract_from_content(&mut first_user_message.content)
            {
                debug!("Directive applied from CLAUDE.md.");
                Self::apply_directive(request, &dir);
            }
        }
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

    fn apply_directive(request: &mut ClaudeMessagesRequest, directive: &ProxyDirective) {
        if let Some(global_settings) = &directive.global {
            Self::apply_settings(request, global_settings);
        }

        for rule in &directive.rules {
            if Self::evaluate_condition(request, &rule.if_clause) {
                Self::apply_settings(request, &rule.apply);
            }
        }
    }

    fn evaluate_condition(request: &ClaudeMessagesRequest, condition: &Condition) -> bool {
        match condition {
            Condition::ModelContains(substring) => request.model.contains(substring),
        }
    }

    fn apply_settings(request: &mut ClaudeMessagesRequest, settings: &Settings) {
        if let Some(model) = &settings.model {
            request.model.clone_from(model);
        }
        if let Some(max_tokens) = settings.max_tokens {
            request.max_tokens = max_tokens;
        }
        if let Some(temp) = settings.temperature {
            request.temperature = Some(temp);
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
}
