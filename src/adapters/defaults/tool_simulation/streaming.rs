use super::parsing::ParsedToolCall;
use memchr::memmem;
use tokio::task::consume_budget;
use tracing::debug;

const YIELD_CHUNK_SIZE: usize = 2048;

#[derive(Debug, Clone)]
pub enum ToolEvent {
    Start { index: usize, name: String },
    Arg { index: usize, delta: String },
    End,
    ToolsBlockEnd,
}

pub trait ToolGrammar: Send {
    fn start_delimiter(&self) -> &'static str;
    fn end_delimiter(&self) -> Option<&'static str>;
    fn extract_one_call_from(&self, buffer: &str, start_at: usize) -> Option<(usize, usize)>;
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

    fn extract_one_call_from(&self, buffer: &str, start_at: usize) -> Option<(usize, usize)> {
        let haystack = &buffer.as_bytes()[start_at..];
        let relative_start = memmem::find(haystack, b"[tool(")?;
        let start_index = start_at + relative_start;
        let bytes = buffer.as_bytes();
        let mut index = start_index + b"[tool(".len();
        let mut in_string = false;
        let mut in_triple_quote = false;
        let mut escape_next = false;
        let is_triple = |i: usize| i + 2 < bytes.len() && &bytes[i..i + 3] == b"\"\"\"";
        while index < bytes.len() {
            let byte = bytes[index];
            if in_triple_quote {
                if is_triple(index) {
                    in_triple_quote = false;
                    index += 3;
                    continue;
                }
                index += 1;
                continue;
            }
            if in_string {
                if !escape_next && byte == b'"' {
                    in_string = false;
                }
                escape_next = !escape_next && byte == b'\\';
                index += 1;
                continue;
            }
            if byte == b'"' {
                if is_triple(index) {
                    in_triple_quote = true;
                    index += 3;
                    continue;
                }
                in_string = true;
                index += 1;
                continue;
            }
            if byte == b')' && index + 1 < bytes.len() && bytes[index + 1] == b']' {
                let end_index = index + 2;
                return Some((start_index, end_index));
            }
            index += 1;
        }
        None
    }

    fn parse_partial_on_finalize(&self, buffer: &str) -> Option<ParsedToolCall> {
        let start = buffer.rfind("[tool(")?;
        let tail = &buffer[start..];
        let candidate = if tail.ends_with(")]") {
            tail.to_string()
        } else {
            format!("{tail})]")
        };
        super::parsing::parse_bracket_tool(&candidate)
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

    fn extract_one_call_from(&self, buffer: &str, start_at: usize) -> Option<(usize, usize)> {
        let haystack = &buffer.as_bytes()[start_at..];
        let start_rel = memmem::find(haystack, b"<invoke ")?;
        let start_index = start_at + start_rel;
        let hay_after = &buffer.as_bytes()[start_index..];
        let end_rel = memmem::find(hay_after, b"</invoke>")?;
        let end_index = start_index + end_rel + b"</invoke>".len();
        Some((start_index, end_index))
    }

    fn parse_partial_on_finalize(&self, buffer: &str) -> Option<ParsedToolCall> {
        let start = buffer.rfind("<invoke ")?;
        let tail = &buffer[start..];
        let candidate = if tail.contains("</invoke>") {
            tail.to_string()
        } else {
            format!("{tail}</invoke>")
        };
        super::parsing::parse_xml_tool(&candidate)
    }
}

#[derive(Debug)]
pub enum ParserState {
    Passthrough,
    MatchingStart { matched_len: usize },
    InToolBlock { buffer: String },
}

#[derive(Default, Debug)]
struct XmlIncState {
    active: bool,
    index: Option<usize>,
    start_index: usize,
    last_emit: usize,
}

#[derive(Default, Debug)]
struct BracketIncState {
    active: bool,
    index: Option<usize>,
    start_index: Option<usize>,
    last_emit: usize,
    phase: BracketPhase,
    string: StringState,
}

#[derive(Default, Debug, Clone, Copy)]
enum BracketPhase {
    #[default]
    SeekingStart,
    SeekingArgs,
    StreamingArgs,
}

#[derive(Default, Debug, Clone, Copy)]
enum StringState {
    #[default]
    None,
    InString {
        escape: bool,
    },
    InTriple,
}

struct StepCtx<'a> {
    out: &'a mut String,
    rel: &'a mut usize,
    reached_end: &'a mut bool,
    end_abs: &'a mut Option<usize>,
}

pub struct StreamingToolParser {
    pub state: ParserState,
    pub grammar: Box<dyn ToolGrammar + Send>,
    pub emitted: usize,
    is_xml: bool,
    xml_state: XmlIncState,
    bracket_state: BracketIncState,
}

impl StreamingToolParser {
    #[must_use]
    pub fn new(grammar: Box<dyn ToolGrammar + Send>) -> Self {
        let is_xml = grammar.start_delimiter() == "<function_calls>";
        Self {
            state: ParserState::Passthrough,
            grammar,
            emitted: 0,
            is_xml,
            xml_state: XmlIncState::default(),
            bracket_state: BracketIncState::default(),
        }
    }

    fn trim_tool_buffer(&mut self, buffer: &mut String) {
        let window = 64usize;
        if self.is_xml {
            if self.xml_state.active {
                let keep_from = self.xml_state.start_index.min(buffer.len());
                if keep_from > 0 {
                    buffer.drain(..keep_from);
                    self.xml_state.last_emit = self.xml_state.last_emit.saturating_sub(keep_from);
                    self.xml_state.start_index = 0;
                }
            } else if buffer.len() > window {
                let keep_from = buffer.len() - window;
                buffer.drain(..keep_from);
            }
        } else if self.bracket_state.active {
            if let Some(start_abs) = self.bracket_state.start_index {
                let keep_from = start_abs.min(buffer.len());
                if keep_from > 0 {
                    buffer.drain(..keep_from);
                    self.bracket_state.last_emit =
                        self.bracket_state.last_emit.saturating_sub(keep_from);
                    if let Some(s) = self.bracket_state.start_index.as_mut() {
                        *s = s.saturating_sub(keep_from);
                    }
                }
            }
        } else if buffer.len() > window {
            let keep_from = buffer.len() - window;
            buffer.drain(..keep_from);
        }
    }

    pub fn reserve_indices(&mut self, count: usize) -> usize {
        let start = self.emitted;
        self.emitted = self.emitted.saturating_add(count);
        start
    }

    fn handle_passthrough_char(
        character: char,
        start_first: Option<char>,
        text_to_yield: &mut String,
    ) -> ParserState {
        if let Some(first) = start_first {
            if character == first {
                return ParserState::MatchingStart { matched_len: 1 };
            }
            text_to_yield.push(character);
            ParserState::Passthrough
        } else {
            text_to_yield.push(character);
            ParserState::Passthrough
        }
    }

    fn handle_matching_start_char(
        character: char,
        matched_len: usize,
        start_delimiter_chars: &[char],
        start_delimiter: &str,
        text_to_yield: &mut String,
    ) -> ParserState {
        if start_delimiter_chars.get(matched_len).copied() == Some(character) {
            let new_length = matched_len + 1;
            if new_length == start_delimiter.len() {
                debug!("tool_sim_enter_block");
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

    fn scan_xml(&mut self, buffer: &mut String, events: &mut Vec<ToolEvent>) {
        if !self.xml_state.active
            && let Some(start_pos) = buffer.find("<invoke ")
            && let Some(name_attr_pos) = buffer[start_pos..].find("name=")
        {
            let name_start = start_pos + name_attr_pos + "name=".len();
            let quote = buffer[name_start..].chars().next();
            if let Some(q) = quote
                && (q == '"' || q == '\'')
                && let Some(end_rel) = buffer[name_start + 1..].find(q)
            {
                let name = buffer[name_start + 1..name_start + 1 + end_rel].to_string();
                let index = self.reserve_indices(1);
                self.xml_state.active = true;
                self.xml_state.index = Some(index);
                self.xml_state.start_index = start_pos;
                self.xml_state.last_emit = start_pos;
                debug!("tool_sim_xml_start index={index} name={name} start={start_pos}");
                events.push(ToolEvent::Start { index, name });
            }
        }

        if self.xml_state.active {
            let mut cursor = self.xml_state.last_emit;
            while let Some(gt_rel) = buffer[cursor..].find('>') {
                let text_start = cursor + gt_rel + 1;
                if let Some(lt_rel) = buffer[text_start..].find('<') {
                    let text_end = text_start + lt_rel;
                    cursor = text_end;
                } else {
                    break;
                }
            }
            self.xml_state.last_emit = cursor;

            if let Some(end_rel) = buffer[self.xml_state.last_emit..].find("</invoke>") {
                let end_index = self.xml_state.last_emit + end_rel + "</invoke>".len();
                if let Some(index) = self.xml_state.index {
                    let start = self.xml_state.start_index;
                    if start < end_index && end_index <= buffer.len() {
                        let slice = &buffer[start..end_index];
                        if let Some(call) = super::parsing::parse_xml_tool(slice) {
                            let args = serde_json::to_string(&call.args).unwrap_or_default();
                            debug!("tool_sim_xml_args index={index} slice={slice} args={args}");
                            events.push(ToolEvent::Arg { index, delta: args });
                        } else {
                            debug!("tool_sim_xml_args_parse_failed index={index} slice={slice}");
                            events.push(ToolEvent::Arg {
                                index,
                                delta: "{}".to_string(),
                            });
                        }
                    } else if start < end_index {
                        events.push(ToolEvent::Arg {
                            index,
                            delta: "{}".to_string(),
                        });
                    }
                    events.push(ToolEvent::End);
                }
                self.xml_state = XmlIncState::default();
                if end_index <= buffer.len() {
                    buffer.drain(..end_index);
                }
            }
        }
    }

    fn triple_next(iter: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> bool {
        if let Some((_, n1)) = iter.peek().copied()
            && n1 == '"'
        {
            let mut it2 = iter.clone();
            let _ = it2.next();
            if let Some((_, n2)) = it2.next()
                && n2 == '"'
            {
                return true;
            }
        }
        false
    }

    fn bracket_step_in_triple(
        &mut self,
        iter: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
        ch: char,
        pos: usize,
        out: &mut String,
        rel: &mut usize,
    ) -> bool {
        if ch == '"' && Self::triple_next(iter) {
            let _ = iter.next();
            let _ = iter.next();
            self.bracket_state.string = StringState::None;
            out.push_str("\"\"\"");
            *rel = pos + 3;
            return false;
        }
        if ch == '"' {
            if iter.peek().is_none() {
                return true;
            }
            out.push(ch);
            *rel = pos + ch.len_utf8();
            return false;
        }
        out.push(ch);
        *rel = pos + ch.len_utf8();
        false
    }

    fn bracket_step_in_string(
        &mut self,
        ch: char,
        pos: usize,
        out: &mut String,
        rel: &mut usize,
        escape: bool,
    ) {
        if escape {
            self.bracket_state.string = StringState::InString { escape: false };
            out.push(ch);
            *rel = pos + ch.len_utf8();
            return;
        }
        if ch == '\\' {
            self.bracket_state.string = StringState::InString { escape: true };
            out.push(ch);
            *rel = pos + ch.len_utf8();
            return;
        }
        if ch == '"' {
            self.bracket_state.string = StringState::None;
            out.push(ch);
            *rel = pos + ch.len_utf8();
            return;
        }
        self.bracket_state.string = StringState::InString { escape: false };
        out.push(ch);
        *rel = pos + ch.len_utf8();
    }

    fn bracket_step_in_none(
        &mut self,
        start: usize,
        iter: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
        ch: char,
        pos: usize,
        ctx: &mut StepCtx,
    ) -> bool {
        if ch == '"' {
            let triple = Self::triple_next(iter);
            if triple {
                self.bracket_state.string = StringState::InTriple;
                ctx.out.push_str("\"\"\"");
                let _ = iter.next();
                let _ = iter.next();
                *ctx.rel = pos + 3;
            } else {
                if iter.peek().is_none() {
                    return true;
                }
                self.bracket_state.string = StringState::InString { escape: false };
                ctx.out.push(ch);
                *ctx.rel = pos + ch.len_utf8();
            }
            return false;
        }
        if ch == ')' {
            match iter.peek().copied() {
                Some((br_pos, ']')) => {
                    *ctx.end_abs = Some(start.saturating_add(br_pos + ']'.len_utf8()));
                    *ctx.reached_end = true;
                    return true;
                }
                Some(_) => {
                    ctx.out.push(ch);
                    *ctx.rel = pos + ch.len_utf8();
                    return false;
                }
                None => {
                    return true;
                }
            }
        }
        ctx.out.push(ch);
        *ctx.rel = pos + ch.len_utf8();
        false
    }

    fn bracket_seek_start(&mut self, buffer: &str, events: &mut Vec<ToolEvent>) -> bool {
        if let Some(start_pos) = buffer.find("[tool(") {
            let name_start = start_pos + "[tool(".len();
            let rest = &buffer[name_start..];
            let Some(end_rel) = rest.find([',', ')']) else {
                return false;
            };
            let name_slice = rest[..end_rel].trim();
            let name: String = name_slice
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if name.is_empty() {
                return false;
            }
            let index = self.reserve_indices(1);
            self.bracket_state.active = true;
            self.bracket_state.index = Some(index);
            self.bracket_state.start_index = Some(start_pos);
            self.bracket_state.last_emit = name_start + end_rel;
            self.bracket_state.phase = BracketPhase::SeekingArgs;
            debug!("tool_sim_bracket_start index={index} name={name} start={start_pos}");
            events.push(ToolEvent::Start { index, name });
            true
        } else {
            false
        }
    }

    fn bracket_seek_args(&mut self, buffer: &mut String, events: &mut Vec<ToolEvent>) -> bool {
        let mut k = self.bracket_state.last_emit;
        while k < buffer.len() {
            let Some(ch) = buffer[k..].chars().next() else {
                break;
            };
            if ch.is_whitespace() {
                k += ch.len_utf8();
                continue;
            }
            if ch == ',' {
                k += ch.len_utf8();
                self.bracket_state.last_emit = k;
                self.bracket_state.phase = BracketPhase::StreamingArgs;
                return true;
            }
            if ch == ')' {
                let rest_index = k + ch.len_utf8();
                if buffer[rest_index..].starts_with(']') {
                    if let Some(index) = self.bracket_state.index {
                        events.push(ToolEvent::Arg {
                            index,
                            delta: "{}".to_string(),
                        });
                        events.push(ToolEvent::End);
                    }
                    let end_index = rest_index + ']'.len_utf8();
                    if end_index <= buffer.len() {
                        buffer.drain(..end_index);
                    } else {
                        buffer.clear();
                    }
                    self.bracket_state = BracketIncState::default();
                    return true;
                }
            }
            break;
        }
        false
    }

    fn bracket_stream_args(&mut self, buffer: &mut String, events: &mut Vec<ToolEvent>) -> bool {
        let mut rel = 0usize;
        let start = self.bracket_state.last_emit.min(buffer.len());
        if start >= buffer.len() {
            return false;
        }
        let slice = &buffer[start..];
        let mut iter = slice.char_indices().peekable();
        let mut reached_end = false;
        let mut end_abs: Option<usize> = None;
        let mut out = String::new();
        while let Some((pos, ch)) = iter.next() {
            match self.bracket_state.string {
                StringState::InTriple => {
                    if self.bracket_step_in_triple(&mut iter, ch, pos, &mut out, &mut rel) {
                        break;
                    }
                }
                StringState::InString { escape } => {
                    self.bracket_step_in_string(ch, pos, &mut out, &mut rel, escape);
                }
                StringState::None => {
                    let mut ctx = StepCtx {
                        out: &mut out,
                        rel: &mut rel,
                        reached_end: &mut reached_end,
                        end_abs: &mut end_abs,
                    };
                    if self.bracket_step_in_none(start, &mut iter, ch, pos, &mut ctx) {
                        break;
                    }
                    if reached_end {
                        break;
                    }
                }
            }
        }
        if !out.is_empty() {
            self.bracket_state.last_emit = self.bracket_state.last_emit.saturating_add(rel);
        }
        if reached_end {
            if let Some(index) = self.bracket_state.index {
                if let (Some(start_abs), Some(end_abs_val)) =
                    (self.bracket_state.start_index, end_abs)
                {
                    if start_abs < end_abs_val && end_abs_val <= buffer.len() {
                        let slice = &buffer[start_abs..end_abs_val];
                        if let Some(call) = super::parsing::parse_bracket_tool(slice) {
                            let args = serde_json::to_string(&call.args).unwrap_or_default();
                            debug!("tool_sim_bracket_args index={index} slice={slice} args={args}");
                            events.push(ToolEvent::Arg { index, delta: args });
                        } else {
                            debug!(
                                "tool_sim_bracket_args_parse_failed index={index} slice={slice}"
                            );
                            events.push(ToolEvent::Arg {
                                index,
                                delta: "{}".to_string(),
                            });
                        }
                        if end_abs_val <= buffer.len() {
                            buffer.drain(..end_abs_val);
                        } else {
                            buffer.clear();
                        }
                    } else {
                        events.push(ToolEvent::Arg {
                            index,
                            delta: "{}".to_string(),
                        });
                    }
                } else {
                    events.push(ToolEvent::Arg {
                        index,
                        delta: "{}".to_string(),
                    });
                }
                events.push(ToolEvent::End);
            }
            self.bracket_state = BracketIncState::default();
            return true;
        }
        false
    }

    fn scan_bracket(&mut self, buffer: &mut String, events: &mut Vec<ToolEvent>) {
        loop {
            match self.bracket_state.phase {
                BracketPhase::SeekingStart => {
                    if !self.bracket_seek_start(buffer.as_str(), events) {
                        return;
                    }
                }
                BracketPhase::SeekingArgs => {
                    if !self.bracket_seek_args(buffer, events) {
                        return;
                    }
                }
                BracketPhase::StreamingArgs => {
                    if !self.bracket_stream_args(buffer, events) {
                        return;
                    }
                }
            }
        }
    }

    fn handle_in_tool_block_char(
        &mut self,
        character: char,
        mut buffer: String,
        end_first: Option<char>,
        end_delim: Option<&str>,
        events: &mut Vec<ToolEvent>,
    ) -> (ParserState, bool) {
        buffer.push(character);

        if self.is_xml {
            self.scan_xml(&mut buffer, events);
        } else {
            self.scan_bracket(&mut buffer, events);
        }
        self.trim_tool_buffer(&mut buffer);

        if let (Some(first), Some(delimiter)) = (end_first, end_delim)
            && character == first
            && buffer.ends_with(delimiter)
        {
            buffer.truncate(buffer.len() - delimiter.len());
            if !self.is_xml {
                if self.bracket_state.last_emit > buffer.len() {
                    self.bracket_state.last_emit = buffer.len();
                }
                self.scan_bracket(&mut buffer, events);
                if self.bracket_state.active {
                    if let (Some(index), Some(start_abs)) =
                        (self.bracket_state.index, self.bracket_state.start_index)
                    {
                        let tail = if start_abs <= buffer.len() {
                            &buffer[start_abs..]
                        } else {
                            ""
                        };
                        let trimmed = tail.trim_end();
                        let candidate = if trimmed.ends_with(")]") {
                            trimmed.to_string()
                        } else {
                            format!("{trimmed})]")
                        };
                        if let Some(call) = super::parsing::parse_bracket_tool(&candidate) {
                            let delta = serde_json::to_string(&call.args).unwrap_or_default();
                            events.push(ToolEvent::Arg { index, delta });
                        } else {
                            events.push(ToolEvent::Arg {
                                index,
                                delta: "{}".to_string(),
                            });
                        }
                        events.push(ToolEvent::End);
                    }
                    self.bracket_state = BracketIncState::default();
                }
            }
            events.push(ToolEvent::ToolsBlockEnd);
            return (ParserState::Passthrough, true);
        }
        (ParserState::InToolBlock { buffer }, false)
    }

    pub async fn process(&mut self, text: &str) -> (String, Vec<ToolEvent>) {
        let mut text_to_yield = String::new();
        let mut events = Vec::new();
        let start_delimiter = self.grammar.start_delimiter();
        let start_delimiter_chars: Vec<char> = start_delimiter.chars().collect();
        let start_first = start_delimiter.chars().next();
        let end_delim = self.grammar.end_delimiter();
        let end_first = end_delim.and_then(|d| d.chars().next());

        let mut processed = 0usize;
        for character in text.chars() {
            let current = std::mem::replace(&mut self.state, ParserState::Passthrough);
            self.state = match current {
                ParserState::Passthrough => {
                    Self::handle_passthrough_char(character, start_first, &mut text_to_yield)
                }
                ParserState::MatchingStart { matched_len } => Self::handle_matching_start_char(
                    character,
                    matched_len,
                    &start_delimiter_chars,
                    start_delimiter,
                    &mut text_to_yield,
                ),
                ParserState::InToolBlock { buffer } => {
                    let (next_state, drained_end) = self.handle_in_tool_block_char(
                        character,
                        buffer,
                        end_first,
                        end_delim,
                        &mut events,
                    );
                    if drained_end {
                        consume_budget().await;
                    }
                    next_state
                }
            };
            processed += 1;
            if processed >= YIELD_CHUNK_SIZE {
                consume_budget().await;
                processed = 0;
            }
        }

        if matches!(self.state, ParserState::InToolBlock { .. }) {
            let mut buffer = match std::mem::replace(&mut self.state, ParserState::Passthrough) {
                ParserState::InToolBlock { buffer } => buffer,
                other => {
                    self.state = other;
                    String::new()
                }
            };
            if self.is_xml {
                self.scan_xml(&mut buffer, &mut events);
            } else {
                self.scan_bracket(&mut buffer, &mut events);
            }
            self.trim_tool_buffer(&mut buffer);
            self.state = ParserState::InToolBlock { buffer };
            consume_budget().await;
        }

        (text_to_yield, events)
    }

    pub async fn finalize(&mut self) -> Vec<ToolEvent> {
        consume_budget().await;
        let mut events = Vec::new();
        if matches!(self.state, ParserState::InToolBlock { .. }) {
            let mut buffer = match std::mem::replace(&mut self.state, ParserState::Passthrough) {
                ParserState::InToolBlock { buffer } => buffer,
                other => {
                    self.state = other;
                    String::new()
                }
            };
            if self.is_xml {
                self.scan_xml(&mut buffer, &mut events);
            } else {
                self.scan_bracket(&mut buffer, &mut events);
            }
            consume_budget().await;
            if !buffer.is_empty()
                && !self.is_xml
                && let Some(call) = self.grammar.parse_partial_on_finalize(&buffer)
            {
                let index = self.reserve_indices(1);
                let args = serde_json::to_string(&call.args).unwrap_or_default();
                events.push(ToolEvent::Arg { index, delta: args });
                events.push(ToolEvent::End);
            }
            self.state = ParserState::InToolBlock { buffer };
        }
        events
    }
}
