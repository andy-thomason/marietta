/// Mandy parser: recursive-descent with a Pratt expression parser.
///
/// The parser never panics.  On unexpected input it emits an error node,
/// skips to a synchronisation point (next `Newline` or `Dedent`), and
/// continues so that as many errors as possible are surfaced in one pass.
///
/// Call [`parse`] to parse a complete source file.
use crate::ast::*;
use crate::lexer::{Lexer, Token};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic<'src> {
    /// The source slice that triggered the diagnostic.
    pub src: &'src str,
    pub message: &'static str,
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

struct Parser<'src> {
    tokens: Vec<Token<'src>>,
    pos: usize,
    pub diagnostics: Vec<Diagnostic<'src>>,
    /// The full source string — needed for constructing zero-length slices.
    source: &'src str,
}

impl<'src> Parser<'src> {
    fn new(source: &'src str) -> Self {
        // Collect all tokens, stripping comments but keeping structure tokens.
        let tokens: Vec<Token<'src>> = Lexer::new(source)
            .filter(|t| !matches!(t, Token::Comment(_)))
            .collect();
        Parser { tokens, pos: 0, diagnostics: Vec::new(), source }
    }

    // ------------------------------------------------------------------
    // Token navigation
    // ------------------------------------------------------------------

    fn peek(&self) -> &Token<'src> {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn peek2(&self) -> &Token<'src> {
        self.tokens.get(self.pos + 1).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token<'src> {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() { self.pos += 1; }
        tok
    }

    /// Skip `Newline` tokens.
    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
    }

    /// Expect and consume a specific punctuation string.  On mismatch, push a
    /// diagnostic and return `false`.
    fn expect_punct(&mut self, s: &'static str) -> bool {
        if matches!(self.peek(), Token::Punctuation(p) if *p == s) {
            self.advance();
            true
        } else {
            let bad = self.current_src();
            self.diagnostics.push(Diagnostic { src: bad, message: "expected punctuation" });
            false
        }
    }

    /// Expect and consume a specific keyword string.
    fn expect_kw(&mut self, s: &'static str) -> bool {
        if matches!(self.peek(), Token::Keyword(k) if *k == s) {
            self.advance();
            true
        } else {
            let bad = self.current_src();
            self.diagnostics.push(Diagnostic { src: bad, message: "expected keyword" });
            false
        }
    }

    /// Advance and return the identifier text, or push a diagnostic.
    /// Also accepts `self` which is classified as a keyword.
    fn expect_ident(&mut self) -> Option<&'src str> {
        match self.peek() {
            Token::Identifier(name) => {
                let name = *name;
                self.advance();
                Some(name)
            }
            Token::Keyword("self") => {
                self.advance();
                Some("self")
            }
            _ => {
                let bad = self.current_src();
                self.diagnostics.push(Diagnostic { src: bad, message: "expected identifier" });
                None
            }
        }
    }

    /// A zero-length slice at the current token position (for synthesised nodes).
    fn current_src(&self) -> &'src str {
        match self.peek() {
            Token::Keyword(s) | Token::Identifier(s) | Token::IntLiteral(s)
            | Token::FloatLiteral(s) | Token::StringLiteral(s)
            | Token::Punctuation(s) | Token::Indent(s) | Token::Comment(s) => s,
            Token::Error(s, _) => s,
            Token::Dedent | Token::Newline | Token::Eof => {
                // Return a zero-length slice at the end of source.
                &self.source[self.source.len()..]
            }
        }
    }

    /// Consume tokens until we hit a `Newline`, `Dedent`, or `Eof`.
    fn synchronise(&mut self) {
        loop {
            match self.peek() {
                Token::Newline | Token::Dedent | Token::Eof => break,
                _ => { self.advance(); }
            }
        }
    }

    /// Return a slice from `start` to the beginning of the current token.
    fn src_from(&self, start: &'src str) -> &'src str {
        let src_start = self.source.as_ptr() as usize;
        let tok_start = start.as_ptr() as usize;
        let tok_end = self.current_src().as_ptr() as usize;
        let begin = tok_start - src_start;
        let end = (tok_end - src_start).max(begin);
        &self.source[begin..end]
    }

    // ------------------------------------------------------------------
    // Type expressions
    // ------------------------------------------------------------------

    fn parse_type_expr(&mut self) -> TypeExpr<'src> {
        let start = self.current_src();

        // Fixed-size array type `[T; N]`
        if matches!(self.peek(), Token::Punctuation("[")) {
            self.advance(); // `[`
            let elem = self.parse_type_expr();
            self.expect_punct(";");
            // Parse the length as an integer literal.
            let len: u64 = if let Token::IntLiteral(s) = self.peek() {
                let n = s.trim_end_matches(|c: char| c == '_' || c.is_alphabetic())
                         .replace('_', "")
                         .parse().unwrap_or(0);
                self.advance();
                n
            } else {
                let bad = self.current_src();
                self.diagnostics.push(Diagnostic { src: bad, message: "expected array length" });
                0
            };
            self.expect_punct("]");
            let src = self.src_from(start);
            return TypeExpr { src, kind: TypeExprKind::Array { elem: Box::new(elem), len } };
        }

        let name = match self.peek() {
            Token::Identifier(s) | Token::Keyword(s) => {
                let s = *s;
                self.advance();
                s
            }
            _ => {
                let bad = self.current_src();
                self.diagnostics.push(Diagnostic { src: bad, message: "expected type name" });
                return TypeExpr { src: bad, kind: TypeExprKind::Error("expected type name") };
            }
        };

        let base = TypeExpr { src: name, kind: TypeExprKind::Name(name) };

        // Generic: `Name[args]`
        if matches!(self.peek(), Token::Punctuation("[")) {
            self.advance();
            let mut args = Vec::new();
            while !matches!(self.peek(), Token::Punctuation("]") | Token::Eof) {
                args.push(self.parse_type_expr());
                if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
            }
            self.expect_punct("]");
            let src = self.src_from(start);
            return TypeExpr {
                src,
                kind: TypeExprKind::Generic { base: Box::new(base), args },
            };
        }

        base
    }

    // ------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------

    fn parse_param_list(&mut self) -> Vec<Param<'src>> {
        self.expect_punct("(");
        let mut params = Vec::new();
        while !matches!(self.peek(), Token::Punctuation(")") | Token::Eof) {
            let start = self.current_src();
            let name = match self.expect_ident() {
                Some(n) => n,
                None => { self.synchronise(); break; }
            };
            let annotation = if matches!(self.peek(), Token::Punctuation(":")) {
                self.advance();
                Some(self.parse_type_expr())
            } else {
                None
            };
            let default = if matches!(self.peek(), Token::Punctuation("=")) {
                self.advance();
                Some(self.parse_expr(0))
            } else {
                None
            };
            let src = self.src_from(start);
            params.push(Param { src, name, annotation, default });
            if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
        }
        self.expect_punct(")");
        params
    }

    // ------------------------------------------------------------------
    // Pratt expression parser
    // ------------------------------------------------------------------

    /// Parse an expression with the given minimum binding power.
    pub fn parse_expr(&mut self, min_bp: u8) -> Expr<'src> {
        let start = self.current_src();
        let mut lhs = self.parse_prefix();

        loop {
            let (l_bp, r_bp) = match self.peek() {
                // Postfix / binary operators — binding powers follow Python
                // precedence (higher = tighter).
                Token::Keyword("if")                      => (2, 1),  // ternary (right-assoc)
                Token::Keyword("or")                      => (4, 5),
                Token::Keyword("and")                     => (6, 7),
                Token::Keyword("not")                     => (0, 8),  // prefix handled below
                Token::Keyword("in") | Token::Keyword("is") => (10, 11),
                Token::Punctuation("<") | Token::Punctuation(">")
                | Token::Punctuation("==") | Token::Punctuation("!=")
                | Token::Punctuation("<=") | Token::Punctuation(">=") => (10, 11),
                Token::Punctuation("|")                   => (12, 13),
                Token::Punctuation("^")                   => (14, 15),
                Token::Punctuation("&")                   => (16, 17),
                Token::Punctuation("<<") | Token::Punctuation(">>") => (18, 19),
                Token::Punctuation("+") | Token::Punctuation("-")   => (20, 21),
                Token::Punctuation("*") | Token::Punctuation("/")
                | Token::Punctuation("//") | Token::Punctuation("%") => (22, 23),
                Token::Punctuation("**")                  => (26, 25), // right-assoc
                // User-defined %…% infix operators — same precedence as *
                Token::Punctuation(s) if s.starts_with('%') && s.ends_with('%') && s.len() > 1 => (22, 23),
                // Postfix
                Token::Punctuation("(")                   => (30, 0),
                Token::Punctuation("[")                   => (30, 0),
                Token::Punctuation(".")                   => (30, 0),
                _ => break,
            };

            if l_bp < min_bp { break; }

            match self.peek() {
                // Ternary `value if cond else alt`
                Token::Keyword("if") => {
                    self.advance();
                    let condition = self.parse_expr(0);
                    self.expect_kw("else");
                    let alt = self.parse_expr(r_bp);
                    let src = self.src_from(start);
                    lhs = Expr {
                        src,
                        kind: ExprKind::IfExpr {
                            condition: Box::new(condition),
                            value: Box::new(lhs),
                            alt: Box::new(alt),
                        },
                    };
                }

                // Call `func(args)`
                Token::Punctuation("(") => {
                    self.advance();
                    let (args, kwargs) = self.parse_call_args();
                    self.expect_punct(")");
                    let src = self.src_from(start);
                    lhs = Expr { src, kind: ExprKind::Call { func: Box::new(lhs), args, kwargs } };
                }

                // Index `obj[idx]`
                Token::Punctuation("[") => {
                    self.advance();
                    let index = self.parse_expr(0);
                    self.expect_punct("]");
                    let src = self.src_from(start);
                    lhs = Expr { src, kind: ExprKind::Index { obj: Box::new(lhs), index: Box::new(index) } };
                }

                // Attribute `obj.name`
                Token::Punctuation(".") => {
                    self.advance();
                    let attr = self.expect_ident().unwrap_or("");
                    let src = self.src_from(start);
                    lhs = Expr { src, kind: ExprKind::Attr { obj: Box::new(lhs), attr } };
                }

                // Binary operator
                _ => {
                    let op = match self.advance() {
                        Token::Punctuation(s) | Token::Keyword(s) => *s,
                        _ => break,
                    };
                    let rhs = self.parse_expr(r_bp);
                    let src = self.src_from(start);
                    lhs = Expr {
                        src,
                        kind: ExprKind::BinOp { op, left: Box::new(lhs), right: Box::new(rhs) },
                    };
                }
            }
        }

        lhs
    }

    fn parse_prefix(&mut self) -> Expr<'src> {
        let start = self.current_src();

        match self.peek().clone() {
            Token::IntLiteral(s) => {
                self.advance();
                Expr { src: s, kind: ExprKind::IntLiteral(s) }
            }
            Token::FloatLiteral(s) => {
                self.advance();
                Expr { src: s, kind: ExprKind::FloatLiteral(s) }
            }
            Token::StringLiteral(s) => {
                self.advance();
                Expr { src: s, kind: ExprKind::StringLiteral(s) }
            }
            Token::Keyword("True") => {
                self.advance();
                Expr { src: start, kind: ExprKind::BoolLiteral(true) }
            }
            Token::Keyword("False") => {
                self.advance();
                Expr { src: start, kind: ExprKind::BoolLiteral(false) }
            }
            Token::Keyword("None") => {
                self.advance();
                Expr { src: start, kind: ExprKind::NoneLiteral }
            }
            Token::Identifier(s) => {
                self.advance();
                Expr { src: s, kind: ExprKind::Name(s) }
            }
            Token::Keyword("self") => {
                let s = start;
                self.advance();
                Expr { src: s, kind: ExprKind::Name("self") }
            }
            // Unary operators
            Token::Punctuation("-") | Token::Punctuation("+")
            | Token::Punctuation("~") => {
                let op = match self.advance() { Token::Punctuation(s) => *s, _ => "-" };
                let operand = self.parse_expr(24); // tighter than binary +/-
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::UnaryOp { op, operand: Box::new(operand) } }
            }
            Token::Keyword("not") => {
                self.advance();
                let operand = self.parse_expr(8);
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::UnaryOp { op: "not", operand: Box::new(operand) } }
            }
            Token::Keyword("await") => {
                self.advance();
                let operand = self.parse_expr(28);
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::Await(Box::new(operand)) }
            }
            Token::Keyword("lambda") => {
                self.advance();
                let params = self.parse_lambda_params();
                self.expect_punct(":");
                let body = self.parse_expr(0);
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::Lambda { params, body: Box::new(body) } }
            }
            // Parenthesised expression or tuple
            Token::Punctuation("(") => {
                self.advance();
                if matches!(self.peek(), Token::Punctuation(")")) {
                    self.advance();
                    return Expr { src: self.src_from(start), kind: ExprKind::Tuple(vec![]) };
                }
                let first = self.parse_expr(0);
                if matches!(self.peek(), Token::Punctuation(",")) {
                    let mut elems = vec![first];
                    while matches!(self.peek(), Token::Punctuation(",")) {
                        self.advance();
                        if matches!(self.peek(), Token::Punctuation(")")) { break; }
                        elems.push(self.parse_expr(0));
                    }
                    self.expect_punct(")");
                    let src = self.src_from(start);
                    Expr { src, kind: ExprKind::Tuple(elems) }
                } else {
                    self.expect_punct(")");
                    first
                }
            }
            // Array / list literal `[a, b, …]`
            Token::Punctuation("[") => {
                self.advance();
                let mut elems = Vec::new();
                while !matches!(self.peek(), Token::Punctuation("]") | Token::Eof) {
                    elems.push(self.parse_expr(0));
                    if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
                }
                self.expect_punct("]");
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::ArrayLit(elems) }
            }
            // Multi-dimensional slice literal `&[start..end, start..end, …]`
            Token::Punctuation("&") if matches!(self.peek2(), Token::Punctuation("[")) => {
                self.advance(); // &
                self.advance(); // [
                let mut ranges = Vec::new();
                while !matches!(self.peek(), Token::Punctuation("]") | Token::Eof) {
                    let range_start = self.parse_expr(0);
                    self.expect_punct("..");
                    let range_end = self.parse_expr(0);
                    ranges.push((range_start, range_end));
                    if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
                }
                self.expect_punct("]");
                let src = self.src_from(start);
                Expr { src, kind: ExprKind::MultiSliceLit(ranges) }
            }
            _ => {
                let bad = self.current_src();
                self.diagnostics.push(Diagnostic { src: bad, message: "expected expression" });
                self.advance();
                Expr { src: bad, kind: ExprKind::Error("expected expression") }
            }
        }
    }

    fn parse_call_args(&mut self) -> (Vec<Expr<'src>>, Vec<(&'src str, Expr<'src>)>) {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        while !matches!(self.peek(), Token::Punctuation(")") | Token::Eof) {
            // keyword argument: name=expr
            if matches!(self.peek(), Token::Identifier(_))
                && matches!(self.peek2(), Token::Punctuation("="))
            {
                let name = match self.advance() { Token::Identifier(s) => *s, _ => "" };
                self.advance(); // '='
                let val = self.parse_expr(0);
                kwargs.push((name, val));
            } else {
                args.push(self.parse_expr(0));
            }
            if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
        }
        (args, kwargs)
    }

    fn parse_lambda_params(&mut self) -> Vec<Param<'src>> {
        let mut params = Vec::new();
        while !matches!(self.peek(), Token::Punctuation(":") | Token::Eof) {
            let start = self.current_src();
            let name = match self.expect_ident() { Some(n) => n, None => break };
            // ':' ends the lambda params — push the final param then stop.
            if matches!(self.peek(), Token::Punctuation(":")) {
                params.push(Param { src: self.src_from(start), name, annotation: None, default: None });
                break;
            }
            let default = if matches!(self.peek(), Token::Punctuation("=")) {
                self.advance();
                Some(self.parse_expr(0))
            } else { None };
            let src = self.src_from(start);
            params.push(Param { src, name, annotation: None, default });
            if matches!(self.peek(), Token::Punctuation(",")) { self.advance(); }
        }
        params
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn parse_block(&mut self) -> Vec<Stmt<'src>> {
        // Expect: Newline Indent stmts Dedent
        self.skip_newlines();
        if !matches!(self.peek(), Token::Indent(_)) {
            let bad = self.current_src();
            self.diagnostics.push(Diagnostic { src: bad, message: "expected indented block" });
            return Vec::new();
        }
        self.advance(); // Indent
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            stmts.push(self.parse_stmt());
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        stmts
    }

    fn parse_stmt(&mut self) -> Stmt<'src> {
        let start = self.current_src();

        let stmt = match self.peek().clone() {
            Token::Keyword("pass") => {
                self.advance();
                Stmt { src: start, kind: StmtKind::Pass }
            }
            Token::Keyword("break") => {
                self.advance();
                Stmt { src: start, kind: StmtKind::Break }
            }
            Token::Keyword("continue") => {
                self.advance();
                Stmt { src: start, kind: StmtKind::Continue }
            }
            Token::Keyword("return") => {
                self.advance();
                let value = if matches!(self.peek(), Token::Newline | Token::Eof | Token::Dedent) {
                    None
                } else {
                    Some(self.parse_expr(0))
                };
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::Return(value) }
            }
            Token::Keyword("var") => {
                self.advance();
                let name = self.expect_ident().unwrap_or("");
                let annotation = if matches!(self.peek(), Token::Punctuation(":")) {
                    self.advance();
                    Some(self.parse_type_expr())
                } else { None };
                let value = if matches!(self.peek(), Token::Punctuation("=")) {
                    self.advance();
                    Some(self.parse_expr(0))
                } else { None };
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::VarDecl { name, annotation, value } }
            }
            Token::Keyword("let") => {
                self.advance();
                let name = self.expect_ident().unwrap_or("");
                let annotation = if matches!(self.peek(), Token::Punctuation(":")) {
                    self.advance();
                    Some(self.parse_type_expr())
                } else { None };
                self.expect_punct("=");
                let value = self.parse_expr(0);
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::LetDecl { name, annotation, value } }
            }
            Token::Keyword("if") => self.parse_if_stmt(),
            Token::Keyword("while") => self.parse_while_stmt(),
            Token::Keyword("for") => self.parse_for_stmt(),
            Token::Keyword("import") => {
                self.advance();
                let module = self.expect_ident().unwrap_or("");
                let alias = if matches!(self.peek(), Token::Keyword("as")) {
                    self.advance();
                    self.expect_ident()
                } else { None };
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::Import { module, alias } }
            }
            Token::Keyword("from") => {
                self.advance();
                let module = self.expect_ident().unwrap_or("");
                self.expect_kw("import");
                let name = self.expect_ident().unwrap_or("");
                let alias = if matches!(self.peek(), Token::Keyword("as")) {
                    self.advance();
                    self.expect_ident()
                } else { None };
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::FromImport { module, name, alias } }
            }
            Token::Error(s, msg) => {
                let s = s;
                let msg = msg;
                self.advance();
                self.diagnostics.push(Diagnostic { src: s, message: msg });
                Stmt { src: s, kind: StmtKind::Error(s, msg) }
            }
            _ => {
                // Could be an assignment or an expression statement.
                let expr = self.parse_expr(0);
                let assign_ops = ["=", "+=", "-=", "*=", "/=", "//=", "%=",
                                  "&=", "|=", "^=", "<<=", ">>=", "**="];
                if let Token::Punctuation(op) = self.peek() {
                    if assign_ops.contains(op) {
                        let op = *op;
                        self.advance();
                        let value = self.parse_expr(0);
                        let src = self.src_from(start);
                        return Stmt { src, kind: StmtKind::Assign { op, target: expr, value } };
                    }
                }
                let src = self.src_from(start);
                Stmt { src, kind: StmtKind::Expr(expr) }
            }
        };

        // Consume trailing newline if present.
        if matches!(self.peek(), Token::Newline) { self.advance(); }
        stmt
    }

    fn parse_if_stmt(&mut self) -> Stmt<'src> {
        let start = self.current_src();
        let mut branches = Vec::new();

        self.expect_kw("if");
        let cond = self.parse_expr(0);
        self.expect_punct(":");
        let body = self.parse_block();
        branches.push((cond, body));

        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Keyword("elif")) {
                self.advance();
                let cond = self.parse_expr(0);
                self.expect_punct(":");
                let body = self.parse_block();
                branches.push((cond, body));
            } else { break; }
        }

        let else_body = if matches!(self.peek(), Token::Keyword("else")) {
            self.advance();
            self.expect_punct(":");
            self.parse_block()
        } else { Vec::new() };

        Stmt { src: self.src_from(start), kind: StmtKind::If { branches, else_body } }
    }

    fn parse_while_stmt(&mut self) -> Stmt<'src> {
        let start = self.current_src();
        self.expect_kw("while");
        let condition = self.parse_expr(0);
        self.expect_punct(":");
        let body = self.parse_block();
        let else_body = if matches!(self.peek(), Token::Keyword("else")) {
            self.advance(); self.expect_punct(":"); self.parse_block()
        } else { Vec::new() };
        Stmt { src: self.src_from(start), kind: StmtKind::While { condition, body, else_body } }
    }

    fn parse_for_stmt(&mut self) -> Stmt<'src> {
        let start = self.current_src();
        self.expect_kw("for");
        // Parse target with min_bp=11 so the `in` keyword (l_bp=10) is not
        // consumed as a binary operator.
        let target = self.parse_expr(11);
        self.expect_kw("in");
        let iter = self.parse_expr(0);
        self.expect_punct(":");
        let body = self.parse_block();
        let else_body = if matches!(self.peek(), Token::Keyword("else")) {
            self.advance(); self.expect_punct(":"); self.parse_block()
        } else { Vec::new() };
        Stmt { src: self.src_from(start), kind: StmtKind::For { target, iter, body, else_body } }
    }

    // ------------------------------------------------------------------
    // Top-level items
    // ------------------------------------------------------------------

    fn parse_function_def(&mut self, is_async: bool) -> FunctionDef<'src> {
        let start = self.current_src();
        let keyword = start;
        self.expect_kw("def");
        let name = self.expect_ident().unwrap_or("");
        let params = self.parse_param_list();
        let return_type = if matches!(self.peek(), Token::Punctuation("->")) {
            self.advance();
            Some(self.parse_type_expr())
        } else { None };
        self.expect_punct(":");
        let body = self.parse_block();
        FunctionDef {
            src: self.src_from(start),
            keyword,
            is_async,
            name,
            params,
            return_type,
            body,
        }
    }

    fn parse_struct_def(&mut self) -> StructDef<'src> {
        let start = self.current_src();
        self.expect_kw("struct");
        let name = self.expect_ident().unwrap_or("");
        self.expect_punct(":");
        self.skip_newlines();
        self.advance(); // Indent
        let mut fields = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            let fstart = self.current_src();
            // Optional `var` keyword
            if matches!(self.peek(), Token::Keyword("var")) { self.advance(); }
            let fname = self.expect_ident().unwrap_or("");
            self.expect_punct(":");
            let annotation = self.parse_type_expr();
            let default = if matches!(self.peek(), Token::Punctuation("=")) {
                self.advance();
                Some(self.parse_expr(0))
            } else { None };
            let fsrc = self.src_from(fstart);
            fields.push(StructField { src: fsrc, name: fname, annotation, default });
            if matches!(self.peek(), Token::Newline) { self.advance(); }
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        StructDef { src: self.src_from(start), name, fields }
    }

    fn parse_impl_block(&mut self) -> ImplBlock<'src> {
        let start = self.current_src();
        self.expect_kw("impl");
        let type_name = self.expect_ident().unwrap_or("");
        self.expect_punct(":");
        self.skip_newlines();
        self.advance(); // Indent
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            let is_async = if matches!(self.peek(), Token::Keyword("async")) {
                self.advance(); true
            } else { false };
            methods.push(self.parse_function_def(is_async));
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        ImplBlock { src: self.src_from(start), type_name, methods }
    }

    fn parse_actor_def(&mut self) -> ActorDef<'src> {
        let start = self.current_src();
        self.expect_kw("actor");
        let name = self.expect_ident().unwrap_or("");
        self.expect_punct(":");
        self.skip_newlines();
        self.advance(); // Indent
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Dedent | Token::Eof) { break; }
            match self.peek().clone() {
                Token::Keyword("var") => {
                    let fstart = self.current_src();
                    self.advance();
                    let fname = self.expect_ident().unwrap_or("");
                    self.expect_punct(":");
                    let annotation = self.parse_type_expr();
                    let default = if matches!(self.peek(), Token::Punctuation("=")) {
                        self.advance(); Some(self.parse_expr(0))
                    } else { None };
                    let fsrc = self.src_from(fstart);
                    fields.push(StructField { src: fsrc, name: fname, annotation, default });
                    if matches!(self.peek(), Token::Newline) { self.advance(); }
                }
                Token::Keyword("async") => {
                    self.advance();
                    methods.push(self.parse_function_def(true));
                }
                Token::Keyword("def") => {
                    methods.push(self.parse_function_def(false));
                }
                _ => { self.synchronise(); }
            }
        }
        if matches!(self.peek(), Token::Dedent) { self.advance(); }
        ActorDef { src: self.src_from(start), name, fields, methods }
    }

    fn parse_item(&mut self) -> Item<'src> {
        self.skip_newlines();
        let start = self.current_src();
        match self.peek().clone() {
            Token::Keyword("async") => {
                self.advance();
                let f = self.parse_function_def(true);
                Item { src: self.src_from(start), kind: ItemKind::FunctionDef(f) }
            }
            Token::Keyword("def") => {
                let f = self.parse_function_def(false);
                Item { src: self.src_from(start), kind: ItemKind::FunctionDef(f) }
            }
            Token::Keyword("struct") => {
                let s = self.parse_struct_def();
                Item { src: self.src_from(start), kind: ItemKind::StructDef(s) }
            }
            Token::Keyword("impl") => {
                let i = self.parse_impl_block();
                Item { src: self.src_from(start), kind: ItemKind::ImplBlock(i) }
            }
            Token::Keyword("actor") => {
                let a = self.parse_actor_def();
                Item { src: self.src_from(start), kind: ItemKind::ActorDef(a) }
            }
            Token::Eof => {
                Item { src: start, kind: ItemKind::Stmt(Stmt { src: start, kind: StmtKind::Pass }) }
            }
            _ => {
                let stmt = self.parse_stmt();
                let src = stmt.src;
                Item { src, kind: ItemKind::Stmt(stmt) }
            }
        }
    }

    fn parse_module(&mut self) -> Module<'src> {
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::Eof) { break; }
            items.push(self.parse_item());
        }
        Module { src: self.source, items }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse `source` into a [`Module`].  Any parse errors are returned alongside
/// the (possibly partial) AST in the [`ParseResult`].
pub struct ParseResult<'src> {
    pub module: Module<'src>,
    pub diagnostics: Vec<Diagnostic<'src>>,
}

pub fn parse(source: &str) -> ParseResult<'_> {
    let mut parser = Parser::new(source);
    let module = parser.parse_module();
    ParseResult { module, diagnostics: parser.diagnostics }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn module(src: &str) -> Module<'_> {
        let r = parse(src);
        // Uncomment to debug: eprintln!("{:?}", r.diagnostics);
        r.module
    }

    fn clean(src: &str) -> ParseResult<'_> {
        parse(src)
    }

    // ---- Expressions -------------------------------------------------------

    #[test]
    fn int_literal() {
        let m = module("42\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::IntLiteral("42"), .. }), .. })
        ));
    }

    #[test]
    fn bool_literals() {
        let m = module("True\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::BoolLiteral(true), .. }), .. })
        ));
    }

    #[test]
    fn string_literal() {
        let m = module("\"hello\"\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::StringLiteral(_), .. }), .. })
        ));
    }

    #[test]
    fn binary_add() {
        let m = module("1 + 2\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::BinOp { op: "+", .. }, .. }), .. })
        ));
    }

    #[test]
    fn binary_precedence() {
        // 1 + 2 * 3  =>  BinOp(+, 1, BinOp(*, 2, 3))
        let m = module("1 + 2 * 3\n");
        if let ItemKind::Stmt(Stmt { kind: StmtKind::Expr(expr), .. }) = &m.items[0].kind {
            if let ExprKind::BinOp { op, right, .. } = &expr.kind {
                assert_eq!(*op, "+");
                assert!(matches!(right.kind, ExprKind::BinOp { op: "*", .. }));
                return;
            }
        }
        panic!("unexpected AST shape");
    }

    #[test]
    fn unary_negate() {
        let m = module("-x\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::UnaryOp { op: "-", .. }, .. }), .. })
        ));
    }

    #[test]
    fn call_expr() {
        let m = module("foo(1, 2)\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::Call { .. }, .. }), .. })
        ));
    }

    #[test]
    fn call_with_kwarg() {
        let m = module("foo(a=1)\n");
        if let ItemKind::Stmt(Stmt { kind: StmtKind::Expr(expr), .. }) = &m.items[0].kind {
            if let ExprKind::Call { kwargs, .. } = &expr.kind {
                assert_eq!(kwargs.len(), 1);
                assert_eq!(kwargs[0].0, "a");
                return;
            }
        }
        panic!("unexpected AST shape");
    }

    #[test]
    fn attr_access() {
        let m = module("obj.field\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::Attr { attr: "field", .. }, .. }), .. })
        ));
    }

    #[test]
    fn index_access() {
        let m = module("arr[0]\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::Index { .. }, .. }), .. })
        ));
    }

    #[test]
    fn ternary_if() {
        let m = module("a if b else c\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::IfExpr { .. }, .. }), .. })
        ));
    }

    #[test]
    fn tuple_literal() {
        let m = module("(1, 2, 3)\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::Tuple(_), .. }), .. })
        ));
    }

    #[test]
    fn list_literal() {
        let m = module("[1, 2, 3]\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::ArrayLit(_), .. }), .. })
        ));
    }

    // ---- Statements --------------------------------------------------------

    #[test]
    fn assign_stmt() {
        let m = module("x = 1\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Assign { op: "=", .. }, .. })
        ));
    }

    #[test]
    fn augmented_assign() {
        let m = module("x += 1\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Assign { op: "+=", .. }, .. })
        ));
    }

    #[test]
    fn var_decl() {
        let m = module("var x: u8 = 10\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::VarDecl { name: "x", .. }, .. })
        ));
    }

    #[test]
    fn let_decl() {
        let m = module("let name: String = \"Mandy\"\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::LetDecl { name: "name", .. }, .. })
        ));
    }

    #[test]
    fn return_stmt() {
        let m = module("def f():\n    return 1\n");
        if let ItemKind::FunctionDef(f) = &m.items[0].kind {
            assert!(matches!(&f.body[0].kind, StmtKind::Return(Some(_))));
        } else { panic!("expected FunctionDef"); }
    }

    #[test]
    fn return_void() {
        let m = module("def f():\n    return\n");
        if let ItemKind::FunctionDef(f) = &m.items[0].kind {
            assert!(matches!(&f.body[0].kind, StmtKind::Return(None)));
        } else { panic!("expected FunctionDef"); }
    }

    #[test]
    fn pass_stmt() {
        let m = module("def f():\n    pass\n");
        if let ItemKind::FunctionDef(f) = &m.items[0].kind {
            assert!(matches!(&f.body[0].kind, StmtKind::Pass));
        } else { panic!("expected FunctionDef"); }
    }

    #[test]
    fn if_stmt() {
        let m = module("if x:\n    pass\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::If { .. }, .. })
        ));
    }

    #[test]
    fn if_else_stmt() {
        let r = clean("if x:\n    pass\nelse:\n    pass\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        assert!(matches!(&r.module.items[0].kind, ItemKind::Stmt(Stmt { kind: StmtKind::If { .. }, .. })));
    }

    #[test]
    fn while_stmt() {
        let m = module("while True:\n    pass\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::While { .. }, .. })
        ));
    }

    #[test]
    fn for_stmt() {
        let m = module("for i in items:\n    pass\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::For { .. }, .. })
        ));
    }

    #[test]
    fn import_stmt() {
        let m = module("import math\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Import { module: "math", alias: None }, .. })
        ));
    }

    // ---- Top-level items ---------------------------------------------------

    #[test]
    fn function_def() {
        let r = clean("def add(a, b):\n    return a + b\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        assert!(matches!(&r.module.items[0].kind, ItemKind::FunctionDef(_)));
    }

    #[test]
    fn function_def_with_annotations() {
        let r = clean("def add(a: u32, b: u32) -> u32:\n    return a + b\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        if let ItemKind::FunctionDef(f) = &r.module.items[0].kind {
            assert_eq!(f.name, "add");
            assert_eq!(f.params.len(), 2);
            assert!(f.return_type.is_some());
        }
    }

    #[test]
    fn async_function_def() {
        let r = clean("async def fetch():\n    pass\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        if let ItemKind::FunctionDef(f) = &r.module.items[0].kind {
            assert!(f.is_async);
        } else { panic!("expected FunctionDef"); }
    }

    #[test]
    fn struct_def() {
        let r = clean("struct Point:\n    x: f32\n    y: f32\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        if let ItemKind::StructDef(s) = &r.module.items[0].kind {
            assert_eq!(s.name, "Point");
            assert_eq!(s.fields.len(), 2);
        } else { panic!("expected StructDef"); }
    }

    #[test]
    fn impl_block() {
        let r = clean("impl Point:\n    def norm(self) -> f32:\n        return 0.0\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        if let ItemKind::ImplBlock(i) = &r.module.items[0].kind {
            assert_eq!(i.type_name, "Point");
            assert_eq!(i.methods.len(), 1);
        } else { panic!("expected ImplBlock"); }
    }

    #[test]
    fn actor_def() {
        let r = clean("actor Counter:\n    var count: u64 = 0\n    def increment(self):\n        self.count += 1\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        if let ItemKind::ActorDef(a) = &r.module.items[0].kind {
            assert_eq!(a.name, "Counter");
            assert_eq!(a.fields.len(), 1);
            assert_eq!(a.methods.len(), 1);
        } else { panic!("expected ActorDef"); }
    }

    // ---- User-defined %…% operators ------------------------------------

    #[test]
    fn percent_op_matmul_parsed() {
        let m = module("a %*% b\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::BinOp { op, .. }, .. }), .. })
            if *op == "%*%"
        ));
    }

    #[test]
    fn percent_op_custom_parsed() {
        let m = module("x %dot% y\n");
        assert!(matches!(
            &m.items[0].kind,
            ItemKind::Stmt(Stmt { kind: StmtKind::Expr(Expr { kind: ExprKind::BinOp { op, .. }, .. }), .. })
            if *op == "%dot%"
        ));
    }

    // ---- Error recovery ----------------------------------------------------

    #[test]
    fn error_in_expr_recovers() {
        // Missing right-hand side — parser should not panic and should emit a diagnostic.
        let r = parse("x = \n");
        assert!(!r.diagnostics.is_empty());
    }

    #[test]
    fn multiple_items_after_error() {
        // An error on one line must not swallow subsequent well-formed items.
        let r = parse("@\ndef f():\n    pass\n");
        assert!(r.module.items.iter().any(|i| matches!(&i.kind, ItemKind::FunctionDef(_))));
    }

    // ---- Source slice identity ---------------------------------------------

    #[test]
    fn src_slice_points_into_source() {
        let src = "x = 42\n";
        let m = module(src);
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            let src_start = src.as_ptr() as usize;
            let stmt_start = stmt.src.as_ptr() as usize;
            assert!(stmt_start >= src_start && stmt_start < src_start + src.len());
        }
    }

    // ---- MultiSlice literal ------------------------------------------------

    #[test]
    fn parse_multislice_1d() {
        let m = module("x = &[10..20]\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::MultiSliceLit(r) if r.len() == 1));
                return;
            }
        }
        panic!("expected MultiSliceLit with 1 range");
    }

    #[test]
    fn parse_multislice_2d() {
        let m = module("x = &[10..20, 30..40]\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::MultiSliceLit(r) if r.len() == 2));
                return;
            }
        }
        panic!("expected MultiSliceLit with 2 ranges");
    }

    #[test]
    fn parse_multislice_no_errors() {
        let r = parse("x = &[10..20, 30..40, 50..60]\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
    }

    // ---- Type suffixes on literals -----------------------------------------

    #[test]
    fn int_suffix_u8() {
        let m = module("x = 42_u8\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::IntLiteral("42_u8")));
                return;
            }
        }
        panic!("expected IntLiteral(\"42_u8\")");
    }

    #[test]
    fn int_suffix_with_separator() {
        let m = module("x = 1_000_u32\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::IntLiteral("1_000_u32")));
                return;
            }
        }
        panic!("expected IntLiteral(\"1_000_u32\")");
    }

    #[test]
    fn float_suffix_f32() {
        let m = module("x = 3.14_f32\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::FloatLiteral("3.14_f32")));
                return;
            }
        }
        panic!("expected FloatLiteral(\"3.14_f32\")");
    }

    // ---- Array literal and type --------------------------------------------

    #[test]
    fn array_literal_is_arraylit() {
        let m = module("x = [1_u8, 2, 3]\n");
        if let ItemKind::Stmt(stmt) = &m.items[0].kind {
            if let StmtKind::Assign { value, .. } = &stmt.kind {
                assert!(matches!(&value.kind, ExprKind::ArrayLit(v) if v.len() == 3));
                return;
            }
        }
        panic!("expected ArrayLit with 3 elements");
    }

    #[test]
    fn array_type_annotation_no_errors() {
        let r = parse("var x: [u8; 10]\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
    }
}
