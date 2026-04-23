/// Tokens produced by the Mandy lexer.
///
/// Every variant that carries a `&'src str` borrows directly from the original
/// source string, so the pointer itself encodes the source position.  No
/// separate `Span` type is needed: you can recover byte offsets with
/// `ptr_sub` when required for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'src> {
    /// A reserved word, e.g. `Keyword("if")`.
    Keyword(&'src str),
    /// Any non-keyword identifier.
    Identifier(&'src str),
    /// An integer literal in decimal, hex (`0x…`), octal (`0o…`), or binary
    /// (`0b…`) form.  The raw source text is kept for later big-int parsing.
    IntLiteral(&'src str),
    /// A floating-point literal, e.g. `3.14` or `1e-10`.
    FloatLiteral(&'src str),
    /// A string literal **including** its surrounding quote characters.
    StringLiteral(&'src str),
    /// An operator or punctuation token, single- or double-character.
    Punctuation(&'src str),
    /// Emitted at the start of every non-empty, non-continuation logical line
    /// to record its indentation.  The slice is the run of spaces at the
    /// beginning of the line; its length is the indent column.
    Indent(&'src str),
    /// Emitted (possibly multiple times) when the indent level decreases.
    Dedent,
    /// End of a logical line (after the last non-whitespace token).
    Newline,
    /// A `#`-comment including the `#` character.
    Comment(&'src str),
    /// A lexer error: the slice points at the offending character(s) and the
    /// `&'static str` is a human-readable message.
    Error(&'src str, &'static str),
    /// End of input.
    Eof,
}

/// Keywords recognised by Mandy.
const KEYWORDS: &[&str] = &[
    "actor", "and", "as", "async", "await", "break", "channel", "class",
    "continue", "def", "elif", "else", "except", "finally", "for", "from",
    "if", "impl", "import", "in", "is", "lambda", "let", "None", "not", "or",
    "pass", "raise", "return", "rpc", "self", "struct", "True", "False",
    "try", "var", "while", "with", "yield",
];

fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// The Mandy lexer.
///
/// Construct it with [`Lexer::new`] and drive it as an [`Iterator`].
pub struct Lexer<'src> {
    /// The full source string.
    src: &'src str,
    /// Byte position of the next character to be consumed.
    pos: usize,
    /// Stack of indentation column widths for open blocks.
    indent_stack: Vec<usize>,
    /// Tokens queued to be yielded before resuming normal scanning.  We use a
    /// small fixed-size array to avoid heap allocation for the common case of
    /// at most two queued tokens (e.g. multiple `Dedent`s + a `Newline`).
    pending: Vec<Token<'src>>,
    /// Whether we are at the very start of a logical line (need to handle
    /// indentation).
    at_line_start: bool,
    /// Set to `true` once we have emitted `Eof`.
    done: bool,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Lexer {
            src,
            pos: 0,
            indent_stack: vec![0],
            pending: Vec::new(),
            at_line_start: true,
            done: false,
        }
    }

    // ------------------------------------------------------------------
    // Low-level character helpers
    // ------------------------------------------------------------------

    fn remaining(&self) -> &'src str {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut chars = self.remaining().chars();
        chars.next();
        chars.next()
    }

    /// Advance by one `char` and return it.
    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Advance while `pred` holds and return the consumed slice.
    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> &'src str {
        let start = self.pos;
        while self.peek().map_or(false, &pred) {
            self.advance();
        }
        &self.src[start..self.pos]
    }

    /// Return a zero-length slice at the current position (for error tokens).
    fn slice_here(&self, len: usize) -> &'src str {
        let end = (self.pos + len).min(self.src.len());
        &self.src[self.pos..end]
    }

    // ------------------------------------------------------------------
    // Indentation handling
    // ------------------------------------------------------------------

    /// Called at the start of each logical line.  Consumes leading spaces,
    /// compares with the indent stack, and pushes `Indent`/`Dedent`/`Newline`
    /// tokens into `self.pending`.
    fn handle_indent(&mut self) {
        let start = self.pos;

        // Count leading spaces; reject tabs immediately.
        loop {
            match self.peek() {
                Some('\t') => {
                    let bad = self.slice_here(1);
                    self.advance();
                    self.pending.push(Token::Error(bad, "tabs are not allowed; use spaces for indentation"));
                    // Treat the tab as 1 space for recovery purposes.
                }
                Some(' ') => {
                    self.advance();
                }
                _ => break,
            }
        }

        let indent_slice = &self.src[start..self.pos];
        let col = indent_slice.len(); // 1 space = 1 column

        // Skip entirely blank lines and comment-only lines without emitting
        // indentation tokens.
        match self.peek() {
            None | Some('\n') | Some('\r') | Some('#') => {
                return;
            }
            _ => {}
        }

        let current = *self.indent_stack.last().unwrap();

        if col > current {
            self.indent_stack.push(col);
            self.pending.push(Token::Indent(indent_slice));
        } else if col < current {
            while *self.indent_stack.last().unwrap() > col {
                self.indent_stack.pop();
                self.pending.push(Token::Dedent);
            }
            if *self.indent_stack.last().unwrap() != col {
                // Inconsistent dedent – report an error using a zero-length
                // slice at the current position.
                let bad = &self.src[start..self.pos];
                self.pending.push(Token::Error(bad, "unindent does not match any outer indentation level"));
            }
        }
        // Equal indentation → no token emitted.
    }

    // ------------------------------------------------------------------
    // Token scanners
    // ------------------------------------------------------------------

    fn scan_comment(&mut self) -> Token<'src> {
        let start = self.pos;
        // consume to end of line (not the newline itself)
        self.take_while(|c| c != '\n' && c != '\r');
        Token::Comment(&self.src[start..self.pos])
    }

    fn scan_string(&mut self, quote: char) -> Token<'src> {
        let start = self.pos;
        self.advance(); // opening quote

        // Triple-quoted strings
        if self.remaining().starts_with([quote, quote].as_slice()) {
            self.advance();
            self.advance();
            loop {
                match self.advance() {
                    None => {
                        return Token::Error(
                            &self.src[start..self.pos],
                            "unterminated triple-quoted string",
                        );
                    }
                    Some('\\') => {
                        self.advance(); // skip escaped char
                    }
                    Some(c) if c == quote => {
                        if self.remaining().starts_with([quote, quote].as_slice()) {
                            self.advance();
                            self.advance();
                            break;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            // Single-quoted string — must end on the same line
            loop {
                match self.advance() {
                    None | Some('\n') | Some('\r') => {
                        return Token::Error(
                            &self.src[start..self.pos],
                            "unterminated string literal",
                        );
                    }
                    Some('\\') => {
                        self.advance(); // skip escaped char
                    }
                    Some(c) if c == quote => break,
                    _ => {}
                }
            }
        }

        // Optional type suffix after the closing quote: `"hello"_bytes`, etc.
        if self.peek() == Some('_') && self.peek2().map_or(false, |c| c.is_alphabetic()) {
            self.advance(); // `_`
            self.take_while(|c| c.is_alphanumeric());
        }

        Token::StringLiteral(&self.src[start..self.pos])
    }

    fn scan_number(&mut self) -> Token<'src> {
        let start = self.pos;

        // Hex / octal / binary prefixes
        if self.peek() == Some('0') {
            match self.peek2() {
                Some('x') | Some('X') => {
                    self.advance(); self.advance(); // 0x
                    self.take_while(|c| c.is_ascii_hexdigit() || c == '_');
                    return Token::IntLiteral(&self.src[start..self.pos]);
                }
                Some('o') | Some('O') => {
                    self.advance(); self.advance(); // 0o
                    self.take_while(|c| matches!(c, '0'..='7') || c == '_');
                    return Token::IntLiteral(&self.src[start..self.pos]);
                }
                Some('b') | Some('B') => {
                    self.advance(); self.advance(); // 0b
                    self.take_while(|c| c == '0' || c == '1' || c == '_');
                    return Token::IntLiteral(&self.src[start..self.pos]);
                }
                _ => {}
            }
        }

        // Decimal integer or float
        self.take_while(|c| c.is_ascii_digit() || c == '_');

        let is_float = match (self.peek(), self.peek2()) {
            (Some('.'), Some(c)) if c.is_ascii_digit() => {
                self.advance(); // '.'
                self.take_while(|c| c.is_ascii_digit() || c == '_');
                true
            }
            (Some('.'), _) if self.peek2().map_or(true, |c| !c.is_alphabetic() && c != '.') => {
                // bare `1.` — treat as float (but not `1..` which is int + range)
                self.advance();
                true
            }
            _ => false,
        };

        // Optional exponent
        let is_float = if matches!(self.peek(), Some('e') | Some('E')) {
            self.advance();
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.advance();
            }
            self.take_while(|c| c.is_ascii_digit() || c == '_');
            true
        } else {
            is_float
        };

        // Optional type suffix: `_u8`, `_f32`, `_i64`, …
        // `take_while` above may have consumed a trailing `_`; if the char
        // just before `self.pos` is `_` and the next char is alphabetic then
        // the `_` is the start of the suffix rather than a digit separator.
        if self.pos > start
            && self.src.as_bytes().get(self.pos - 1) == Some(&b'_')
            && self.peek().map_or(false, |c| c.is_alphabetic())
        {
            self.pos -= 1; // give back the `_`
        }
        if self.peek() == Some('_') && self.peek2().map_or(false, |c| c.is_alphabetic()) {
            self.advance(); // `_`
            self.take_while(|c| c.is_alphanumeric());
        }

        let slice = &self.src[start..self.pos];
        if is_float {
            Token::FloatLiteral(slice)
        } else {
            Token::IntLiteral(slice)
        }
    }

    fn scan_ident_or_keyword(&mut self) -> Token<'src> {
        let start = self.pos;
        self.take_while(|c| c.is_alphanumeric() || c == '_');
        let slice = &self.src[start..self.pos];
        if is_keyword(slice) {
            Token::Keyword(slice)
        } else {
            Token::Identifier(slice)
        }
    }

    /// Scan a punctuation / operator token.  We try two-character operators
    /// first, then fall back to single-character.
    /// Scan a `%…%` user-defined infix operator token.
    /// The opening `%` has been peeked but not consumed.
    fn scan_percent_op(&mut self) -> Token<'src> {
        let start = self.pos;
        self.advance(); // opening '%'
        // Consume the operator name (anything except '%', newline, or EOF)
        self.take_while(|c| c != '%' && c != '\n' && c != '\r');
        if self.peek() == Some('%') {
            self.advance(); // closing '%'
            Token::Punctuation(&self.src[start..self.pos])
        } else {
            // Unterminated — treat the bare '%' as a regular punctuation token.
            Token::Punctuation(&self.src[start..start + 1])
        }
    }

    fn scan_punctuation(&mut self) -> Token<'src> {
        let start = self.pos;
        let two = &self.src[self.pos..self.src.len().min(self.pos + 2)];

        let len = match two {
            "==" | "!=" | "<=" | ">=" | "**" | "//" | "<<" | ">>" | "->"
            | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^="
            | "**=" | "//=" | "<<=" | ">>=" | "::" | ".." => 2,
            _ => {
                match self.peek() {
                    Some(c @ ('+' | '-' | '*' | '/' | '%' | '&' | '|' | '^'
                             | '~' | '<' | '>' | '=' | '!' | '(' | ')'
                             | '[' | ']' | '{' | '}' | ':' | ';' | ','
                             | '.' | '@')) => c.len_utf8(),
                    _ => 0,
                }
            }
        };

        if len == 0 {
            // Unknown character — emit an error token.
            let bad = self.slice_here(1);
            self.advance();
            return Token::Error(bad, "unexpected character");
        }

        self.pos += len;
        Token::Punctuation(&self.src[start..self.pos])
    }

    // ------------------------------------------------------------------
    // Core scan step
    // ------------------------------------------------------------------

    fn next_token(&mut self) -> Token<'src> {
        // Flush any queued tokens (from indent/dedent processing).
        if !self.pending.is_empty() {
            return self.pending.remove(0);
        }

        // Handle indentation at the start of a new line.
        if self.at_line_start {
            self.at_line_start = false;
            self.handle_indent();
            if !self.pending.is_empty() {
                return self.pending.remove(0);
            }
        }

        // Skip inline whitespace (spaces only; tabs already rejected above).
        self.take_while(|c| c == ' ');

        match self.peek() {
            None => {
                // Emit pending Dedents before Eof.
                while self.indent_stack.len() > 1 {
                    self.indent_stack.pop();
                    self.pending.push(Token::Dedent);
                }
                if !self.pending.is_empty() {
                    return self.pending.remove(0);
                }
                Token::Eof
            }

            // Line endings
            Some('\r') => {
                self.advance();
                if self.peek() == Some('\n') { self.advance(); }
                self.at_line_start = true;
                Token::Newline
            }
            Some('\n') => {
                self.advance();
                self.at_line_start = true;
                Token::Newline
            }

            // Line continuation
            Some('\\') => {
                self.advance();
                if matches!(self.peek(), Some('\n') | Some('\r')) {
                    self.advance();
                    if self.peek() == Some('\n') { self.advance(); }
                    // Don't treat the next physical line start as a new logical line.
                    self.take_while(|c| c == ' ');
                    self.next_token()
                } else {
                    Token::Error(self.slice_here(1), "unexpected backslash")
                }
            }

            Some('#') => self.scan_comment(),

            // User-defined infix operators: %name% (e.g. %*%, %+%, %my_op%)
            Some('%') => self.scan_percent_op(),

            Some(q @ ('"' | '\'')) => self.scan_string(q),

            Some(c) if c.is_ascii_digit() => self.scan_number(),

            Some(c) if c.is_alphabetic() || c == '_' => self.scan_ident_or_keyword(),

            Some(_) => self.scan_punctuation(),
        }
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Token<'src>;

    fn next(&mut self) -> Option<Token<'src>> {
        if self.done {
            return None;
        }
        let tok = self.next_token();
        if tok == Token::Eof {
            self.done = true;
        }
        Some(tok)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<Token<'_>> {
        Lexer::new(src).collect()
    }

    fn lex_no_eof(src: &str) -> Vec<Token<'_>> {
        Lexer::new(src).filter(|t| *t != Token::Eof).collect()
    }

    // ---- Keywords --------------------------------------------------------

    #[test]
    fn keywords() {
        let tokens = lex_no_eof("def if else return");
        assert_eq!(tokens, vec![
            Token::Keyword("def"),
            Token::Keyword("if"),
            Token::Keyword("else"),
            Token::Keyword("return"),
        ]);
    }

    // ---- Identifiers -----------------------------------------------------

    #[test]
    fn identifiers() {
        let tokens = lex_no_eof("foo bar_baz _private CamelCase");
        assert_eq!(tokens, vec![
            Token::Identifier("foo"),
            Token::Identifier("bar_baz"),
            Token::Identifier("_private"),
            Token::Identifier("CamelCase"),
        ]);
    }

    // ---- Integer literals ------------------------------------------------

    #[test]
    fn int_decimal() {
        assert_eq!(lex_no_eof("42"), vec![Token::IntLiteral("42")]);
    }

    #[test]
    fn int_hex() {
        assert_eq!(lex_no_eof("0xFF"), vec![Token::IntLiteral("0xFF")]);
    }

    #[test]
    fn int_octal() {
        assert_eq!(lex_no_eof("0o77"), vec![Token::IntLiteral("0o77")]);
    }

    #[test]
    fn int_binary() {
        assert_eq!(lex_no_eof("0b1010"), vec![Token::IntLiteral("0b1010")]);
    }

    #[test]
    fn int_underscores() {
        assert_eq!(lex_no_eof("1_000_000"), vec![Token::IntLiteral("1_000_000")]);
    }

    // ---- Float literals --------------------------------------------------

    #[test]
    fn float_basic() {
        assert_eq!(lex_no_eof("3.14"), vec![Token::FloatLiteral("3.14")]);
    }

    #[test]
    fn float_exponent() {
        assert_eq!(lex_no_eof("1e10"), vec![Token::FloatLiteral("1e10")]);
    }

    #[test]
    fn float_negative_exponent() {
        assert_eq!(lex_no_eof("1.5e-3"), vec![Token::FloatLiteral("1.5e-3")]);
    }

    // ---- String literals -------------------------------------------------

    #[test]
    fn string_double_quote() {
        let src = r#""hello""#;
        assert_eq!(lex_no_eof(src), vec![Token::StringLiteral(src)]);
    }

    #[test]
    fn string_single_quote() {
        let src = "'world'";
        assert_eq!(lex_no_eof(src), vec![Token::StringLiteral(src)]);
    }

    #[test]
    fn string_escape() {
        let src = r#""line\n""#;
        assert_eq!(lex_no_eof(src), vec![Token::StringLiteral(src)]);
    }

    #[test]
    fn string_triple_double() {
        let src = r#""""hello world""""#;
        assert_eq!(lex_no_eof(src), vec![Token::StringLiteral(src)]);
    }

    #[test]
    fn string_unterminated() {
        let tokens = lex_no_eof("\"hello");
        assert!(matches!(tokens[0], Token::Error(_, _)));
    }

    // ---- Comments --------------------------------------------------------

    #[test]
    fn comment() {
        let src = "# this is a comment";
        assert_eq!(lex_no_eof(src), vec![Token::Comment(src)]);
    }

    // ---- Punctuation -----------------------------------------------------

    #[test]
    fn punctuation_single() {
        let src = "+-*/";
        assert_eq!(lex_no_eof(src), vec![
            Token::Punctuation("+"),
            Token::Punctuation("-"),
            Token::Punctuation("*"),
            Token::Punctuation("/"),
        ]);
    }

    #[test]
    fn punctuation_double() {
        let src = "== != <= >=";
        assert_eq!(lex_no_eof(src), vec![
            Token::Punctuation("=="),
            Token::Punctuation("!="),
            Token::Punctuation("<="),
            Token::Punctuation(">="),
        ]);
    }

    #[test]
    fn arrow() {
        assert_eq!(lex_no_eof("->"), vec![Token::Punctuation("->")]);
    }

    // ---- Indentation -----------------------------------------------------

    #[test]
    fn indent_and_dedent() {
        let src = "if True:\n    pass\n";
        let tokens: Vec<_> = lex_no_eof(src);
        // Should contain: Keyword("if"), Keyword("True"), Punctuation(":"),
        //   Newline, Indent("    "), Keyword("pass"), Newline, Dedent
        assert!(tokens.contains(&Token::Indent("    ")));
        assert!(tokens.contains(&Token::Dedent));
    }

    #[test]
    fn no_indent_on_blank_line() {
        // A blank line between two top-level statements must not emit Indent.
        let src = "a\n\nb\n";
        let tokens = lex_no_eof(src);
        assert!(!tokens.iter().any(|t| matches!(t, Token::Indent(_))));
    }

    // ---- User-defined %…% operators ------------------------------------

    #[test]
    fn percent_op_matmul() {
        assert_eq!(lex_no_eof("%*%"), vec![Token::Punctuation("%*%")]);
    }

    #[test]
    fn percent_op_custom() {
        assert_eq!(lex_no_eof("%my_op%"), vec![Token::Punctuation("%my_op%")]);
    }

    #[test]
    fn percent_op_in_expr() {
        let tokens = lex_no_eof("a %*% b");
        assert_eq!(tokens, vec![
            Token::Identifier("a"),
            Token::Punctuation("%*%"),
            Token::Identifier("b"),
        ]);
    }

    #[test]
    fn percent_op_unterminated_falls_back_to_mod() {
        // A lone '%' with no closing '%' becomes a plain '%' punctuation token.
        let tokens = lex_no_eof("%");
        assert_eq!(tokens, vec![Token::Punctuation("%")]);
    }

    // ---- Tab rejection ---------------------------------------------------

    #[test]
    fn tab_produces_error() {
        let src = "if True:\n\tpass\n";
        let tokens: Vec<_> = lex(src);
        assert!(tokens.iter().any(|t| matches!(t, Token::Error(_, msg) if msg.contains("tab"))));
    }

    // ---- Source correspondence ------------------------------------------

    #[test]
    fn slice_points_into_source() {
        let src = "hello = 42";
        let mut lex = Lexer::new(src);
        let first = lex.next().unwrap();
        if let Token::Identifier(s) = first {
            // The slice must be a sub-slice of the original source.
            let src_start = src.as_ptr() as usize;
            let tok_start = s.as_ptr() as usize;
            assert!(tok_start >= src_start && tok_start < src_start + src.len());
        } else {
            panic!("expected Identifier, got {:?}", first);
        }
    }

    // ---- Newline ---------------------------------------------------------

    #[test]
    fn newline_token() {
        let tokens = lex("a\nb");
        assert!(tokens.contains(&Token::Newline));
    }

    // ---- Eof -------------------------------------------------------------

    #[test]
    fn eof_is_last() {
        let tokens = lex("x");
        assert_eq!(tokens.last(), Some(&Token::Eof));
    }

    // ---- Real snippet ----------------------------------------------------

    #[test]
    fn function_def() {
        let src = "def add(a, b):\n    return a + b\n";
        let tokens = lex_no_eof(src);
        assert_eq!(tokens[0], Token::Keyword("def"));
        assert_eq!(tokens[1], Token::Identifier("add"));
    }
}
