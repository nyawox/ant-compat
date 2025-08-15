use super::parsing::ParsedToolCall;

pub trait ToolGrammar: Send {
    fn start_delimiter(&self) -> &'static str;
    fn end_delimiter(&self) -> Option<&'static str>;
    fn extract_one_call(&self, buffer: &str) -> Option<(usize, usize)>;
    fn parse_single_call(&self, slice: &str) -> Option<ParsedToolCall>;
    fn parse_partial_on_finalize(&self, buffer: &str) -> Option<ParsedToolCall>;
}

pub struct BracketGrammar;

impl ToolGrammar for BracketGrammar {
    fn start_delimiter(&self) -> &'static str {
        "---TOOLS---"
    }
    fn end_delimiter(&self) -> Option<&'static str> {
        Some("---END_TOOLS---")
    }

    fn extract_one_call(&self, buffer: &str) -> Option<(usize, usize)> {
        let start = buffer.find("[tool(")?;
        let mut index = start + "[tool(".len();
        let bytes = buffer.as_bytes();
        let mut in_str = false;
        let mut in_triple = false;
        let mut escape = false;
        while index < bytes.len() {
            let character = bytes[index] as char;
            if in_triple {
                if index + 2 < bytes.len() && &buffer[index..index + 3] == r#"""""# {
                    in_triple = false;
                    index += 3;
                    continue;
                }
                index += 1;
                continue;
            }
            if in_str {
                if !escape && character == '"' {
                    in_str = false;
                }
                escape = !escape && character == '\\';
                index += 1;
                continue;
            }
            if character == '"' {
                if index + 2 < bytes.len() && &buffer[index..index + 3] == r#"""""# {
                    in_triple = true;
                    index += 3;
                    continue;
                }
                in_str = true;
                index += 1;
                continue;
            }
            if character == ')' && index + 1 < bytes.len() && bytes[index + 1] as char == ']' {
                let end = index + 2;
                return Some((start, end));
            }
            index += 1;
        }
        None
    }

    fn parse_single_call(&self, slice: &str) -> Option<ParsedToolCall> {
        super::parsing::parse_single_bracket_tool_call(slice)
    }

    fn parse_partial_on_finalize(&self, buffer: &str) -> Option<ParsedToolCall> {
        let start = buffer.rfind("[tool(")?;
        let tail = &buffer[start..];
        let candidate = if tail.ends_with(")]") {
            tail.to_string()
        } else {
            format!("{tail})]")
        };
        self.parse_single_call(&candidate)
    }
}

pub struct XmlGrammar;

impl ToolGrammar for XmlGrammar {
    fn start_delimiter(&self) -> &'static str {
        "<function_calls>"
    }
    fn end_delimiter(&self) -> Option<&'static str> {
        Some("</function_calls>")
    }

    fn extract_one_call(&self, buffer: &str) -> Option<(usize, usize)> {
        let start = buffer.find("<invoke ")?;
        let end_rel = buffer[start..].find("</invoke>")?;
        let end = start + end_rel + "</invoke>".len();
        Some((start, end))
    }

    fn parse_single_call(&self, slice: &str) -> Option<ParsedToolCall> {
        super::parsing::parse_single_xml_tool_call(slice)
    }

    fn parse_partial_on_finalize(&self, buffer: &str) -> Option<ParsedToolCall> {
        let start = buffer.rfind("<invoke ")?;
        let tail = &buffer[start..];
        let candidate = if tail.contains("</invoke>") {
            tail.to_string()
        } else {
            format!("{tail}</invoke>")
        };
        self.parse_single_call(&candidate)
    }
}

#[derive(Debug)]
pub enum ParserState {
    Passthrough,
    MatchingStart { matched_len: usize },
    InToolBlock { buffer: String },
}

pub struct StreamingToolParser {
    pub state: ParserState,
    pub grammar: Box<dyn ToolGrammar + Send>,
    pub emitted: usize,
}

impl StreamingToolParser {
    #[must_use]
    pub fn new(grammar: Box<dyn ToolGrammar + Send>) -> Self {
        Self {
            state: ParserState::Passthrough,
            grammar,
            emitted: 0,
        }
    }

    pub fn reserve_indices(&mut self, count: usize) -> usize {
        let start = self.emitted;
        self.emitted = self.emitted.saturating_add(count);
        start
    }

    pub fn process(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        let mut text_to_yield = String::new();
        let mut tools_to_yield = Vec::new();
        let start_delimiter = self.grammar.start_delimiter();
        let end_delimiter_first_char = self
            .grammar
            .end_delimiter()
            .and_then(|delimiter| delimiter.chars().next());

        for character in text.chars() {
            let current = std::mem::replace(&mut self.state, ParserState::Passthrough);
            self.state = match current {
                ParserState::Passthrough => {
                    if let Some(first) = start_delimiter.chars().next() {
                        if character == first {
                            ParserState::MatchingStart { matched_len: 1 }
                        } else {
                            text_to_yield.push(character);
                            ParserState::Passthrough
                        }
                    } else {
                        text_to_yield.push(character);
                        ParserState::Passthrough
                    }
                }
                ParserState::MatchingStart { matched_len } => {
                    if start_delimiter.chars().nth(matched_len) == Some(character) {
                        let new_length = matched_len + 1;
                        if new_length == start_delimiter.len() {
                            ParserState::InToolBlock {
                                buffer: String::new(),
                            }
                        } else {
                            ParserState::MatchingStart {
                                matched_len: new_length,
                            }
                        }
                    } else {
                        text_to_yield.push_str(&start_delimiter[..matched_len]);
                        text_to_yield.push(character);
                        ParserState::Passthrough
                    }
                }
                ParserState::InToolBlock { mut buffer } => {
                    buffer.push(character);
                    drain_complete_calls(&*self.grammar, &mut buffer, &mut tools_to_yield);
                    if let (Some(first), Some(delimiter)) =
                        (end_delimiter_first_char, self.grammar.end_delimiter())
                    {
                        if character == first && buffer.ends_with(delimiter) {
                            drain_complete_calls(&*self.grammar, &mut buffer, &mut tools_to_yield);
                            ParserState::Passthrough
                        } else {
                            ParserState::InToolBlock { buffer }
                        }
                    } else {
                        ParserState::InToolBlock { buffer }
                    }
                }
            };
        }

        (text_to_yield, tools_to_yield)
    }

    pub fn finalize(&mut self) -> Vec<ParsedToolCall> {
        let mut calls = Vec::new();
        if let ParserState::InToolBlock { buffer } = &mut self.state {
            drain_complete_calls(&*self.grammar, buffer, &mut calls);
            if !buffer.is_empty()
                && let Some(call) = self.grammar.parse_partial_on_finalize(buffer)
            {
                calls.push(call);
            }
        }
        calls
    }
}

fn drain_complete_calls(
    grammar: &dyn ToolGrammar,
    buffer: &mut String,
    out: &mut Vec<ParsedToolCall>,
) {
    loop {
        let Some((start, end)) = grammar.extract_one_call(buffer) else {
            break;
        };
        let slice = &buffer[start..end];
        if let Some(call) = grammar.parse_single_call(slice) {
            out.push(call);
        }
        buffer.replace_range(start..end, "");
    }
}
