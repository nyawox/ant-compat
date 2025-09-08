use bstr::ByteSlice;

#[derive(Debug, Default)]
pub enum ThinkParserState {
    #[default]
    Passthrough,
    Disabled,
}

#[derive(Debug)]
pub struct ThinkTagParser {
    pub state: ThinkParserState,
    buffer: String,
    reentry_disabled: bool,
}

impl Default for ThinkTagParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkTagParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: ThinkParserState::Passthrough,
            buffer: String::new(),
            reentry_disabled: std::env::var("ENABLE_REASONING_REENTRY").is_err(),
        }
    }

    pub fn on_reasoning_mode(&mut self) {
        self.state = ThinkParserState::Disabled;
    }

    pub fn on_think_end(&mut self) {
        if self.reentry_disabled {
            self.state = ThinkParserState::Disabled;
        } else {
            self.state = ThinkParserState::Passthrough;
        }
    }

    #[must_use]
    pub fn is_thinking_allowed(&self) -> bool {
        matches!(self.state, ThinkParserState::Passthrough)
    }

    fn compute_keep_len(bytes: &[u8]) -> usize {
        let tags: [&[u8]; 5] = [b"<think>", b"</think>", b"<cot>", b"</cot>", b"<end_cot>"];
        let max_tag_len = tags.iter().map(|t| t.len()).max().unwrap_or(0);
        let start_index = bytes.len().saturating_sub(max_tag_len.saturating_sub(1));
        let window = &bytes[start_index..];
        if let Some(rel) = window.rfind(b"<") {
            let suffix = &bytes[start_index + rel..];
            let is_prefix = tags.iter().any(|tag| tag.starts_with(suffix));
            let is_full = tags.contains(&suffix);
            if is_prefix && !is_full {
                return suffix.len();
            }
        }
        0
    }

    #[must_use]
    pub fn preprocess(&mut self, input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }

        let combined = if self.buffer.is_empty() {
            input.to_string()
        } else {
            let mut merged = String::with_capacity(self.buffer.len() + input.len());
            merged.push_str(&self.buffer);
            merged.push_str(input);
            merged
        };

        let bytes = combined.as_bytes();
        let keep_tag = Self::compute_keep_len(bytes);

        let mut output_end = bytes.len().saturating_sub(keep_tag);
        while output_end < bytes.len() && !combined.is_char_boundary(output_end) {
            output_end = output_end.saturating_sub(1);
        }

        let output = combined[..output_end].to_string();
        if keep_tag > 0 && output.trim().is_empty() {
            self.buffer = combined;
            return String::new();
        }
        self.buffer = combined[output_end..].to_string();
        output
    }

    #[must_use]
    pub fn clean_before(&self, input: &str) -> String {
        if matches!(self.state, ThinkParserState::Disabled) {
            return input.to_string();
        }
        let mut rest = input;
        loop {
            let trimmed = rest.trim_start();
            if trimmed.len() != rest.len() {
                rest = trimmed;
                continue;
            }
            if rest.starts_with("<think>") {
                rest = &rest["<think>".len()..];
                continue;
            }
            if rest.starts_with("<cot>") {
                rest = &rest["<cot>".len()..];
                continue;
            }
            break;
        }
        rest.to_string()
    }

    #[must_use]
    pub fn clean_after(&self, input: &str) -> String {
        if matches!(self.state, ThinkParserState::Disabled) {
            return input.to_string();
        }
        let mut rest = input;
        loop {
            let trimmed = rest.trim_start();
            if trimmed.len() != rest.len() {
                rest = trimmed;
                continue;
            }
            if rest.starts_with("</think>") {
                rest = &rest["</think>".len()..];
                continue;
            }
            if rest.starts_with("</cot>") {
                rest = &rest["</cot>".len()..];
                continue;
            }
            if rest.starts_with("<end_cot>") {
                rest = &rest["<end_cot>".len()..];
                continue;
            }
            break;
        }
        rest.to_string()
    }
}
