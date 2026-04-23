/// The Marietta AST.
///
/// Every node carries two source slices borrowed from the original source `&str`:
///
/// * `src` — the full text of the production (start of first token to end of
///   last token), used for error messages and source correspondence.
/// * A secondary slice on the "distinguishing" token where useful (e.g. the
///   operator in a `BinOp`, the keyword in a `FunctionDef`).
///
/// Because all slices come from the same underlying `&'src str` their pointers
/// give exact byte offsets with no extra `Span` bookkeeping.

// ---------------------------------------------------------------------------
// Type annotations
// ---------------------------------------------------------------------------

/// A parsed type annotation such as `u8`, `String`, `f32`, or `Vec[u8]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeExpr<'src> {
    pub src: &'src str,
    pub kind: TypeExprKind<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExprKind<'src> {
    /// A simple name like `u8` or `String`.
    Name(&'src str),
    /// A generic like `Vec[u8]`.
    Generic {
        base: Box<TypeExpr<'src>>,
        args: Vec<TypeExpr<'src>>,
    },
    /// A fixed-size array type `[T; N]`.
    Array {
        elem: Box<TypeExpr<'src>>,
        len:  u64,
    },
    /// A parse error in a type position.
    Error(&'static str),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Expr<'src> {
    /// Full source text of this expression.
    pub src: &'src str,
    pub kind: ExprKind<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind<'src> {
    /// Integer literal — raw text for later big-int parsing.
    IntLiteral(&'src str),
    /// Float literal.
    FloatLiteral(&'src str),
    /// String literal including surrounding quotes.
    StringLiteral(&'src str),
    /// `True` or `False`.
    BoolLiteral(bool),
    /// `None`.
    NoneLiteral,
    /// A plain name / identifier.
    Name(&'src str),
    /// `left op right`; `op` is the operator source slice.
    BinOp {
        op: &'src str,
        left: Box<Expr<'src>>,
        right: Box<Expr<'src>>,
    },
    /// `op operand`; `op` is the operator source slice.
    UnaryOp {
        op: &'src str,
        operand: Box<Expr<'src>>,
    },
    /// `func(args, keyword=value, …)`
    Call {
        func: Box<Expr<'src>>,
        args: Vec<Expr<'src>>,
        kwargs: Vec<(&'src str, Expr<'src>)>,
    },
    /// `obj[index]`
    Index {
        obj: Box<Expr<'src>>,
        index: Box<Expr<'src>>,
    },
    /// `obj.attr`; `attr` is the attribute name slice.
    Attr {
        obj: Box<Expr<'src>>,
        attr: &'src str,
    },
    /// `await expr`
    Await(Box<Expr<'src>>),
    /// `value if condition else alt`
    IfExpr {
        condition: Box<Expr<'src>>,
        value: Box<Expr<'src>>,
        alt: Box<Expr<'src>>,
    },
    /// `lambda params: body`
    Lambda {
        params: Vec<Param<'src>>,
        body: Box<Expr<'src>>,
    },
    /// A tuple `(a, b, …)`.
    Tuple(Vec<Expr<'src>>),
    /// A list `[a, b, …]`.
    List(Vec<Expr<'src>>),
    /// A fixed-size array constant `[a, b, …]`.
    ///
    /// Produced by `[expr, expr, …]` in expression position. Unlike `List`
    /// this compiles to a stack-allocated array with a known element count.
    ArrayLit(Vec<Expr<'src>>),
    /// A multi-dimensional slice literal `&[start..end, start..end, …]`.
    ///
    /// Each element is `(start, end)` corresponding to one dimension's range.
    /// The number of pairs is the rank.
    MultiSliceLit(Vec<(Expr<'src>, Expr<'src>)>),
    /// A parse error in expression position.
    Error(&'static str),
}

// ---------------------------------------------------------------------------
// Function parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Param<'src> {
    pub src: &'src str,
    pub name: &'src str,
    pub annotation: Option<TypeExpr<'src>>,
    pub default: Option<Expr<'src>>,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt<'src> {
    /// Full source text of this statement.
    pub src: &'src str,
    pub kind: StmtKind<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind<'src> {
    /// `target = value` or augmented `target op= value`.
    Assign {
        /// The assignment operator (`=`, `+=`, `-=`, etc.).
        op: &'src str,
        target: Expr<'src>,
        value: Expr<'src>,
    },
    /// `var name: type = value`
    VarDecl {
        name: &'src str,
        annotation: Option<TypeExpr<'src>>,
        value: Option<Expr<'src>>,
    },
    /// `let name: type = value`
    LetDecl {
        name: &'src str,
        annotation: Option<TypeExpr<'src>>,
        value: Expr<'src>,
    },
    /// `return [value]`
    Return(Option<Expr<'src>>),
    /// `if cond: … [elif cond: …]* [else: …]`
    If {
        branches: Vec<(Expr<'src>, Vec<Stmt<'src>>)>,
        else_body: Vec<Stmt<'src>>,
    },
    /// `while cond: body [else: …]`
    While {
        condition: Expr<'src>,
        body: Vec<Stmt<'src>>,
        else_body: Vec<Stmt<'src>>,
    },
    /// `for target in iter: body [else: …]`
    For {
        target: Expr<'src>,
        iter: Expr<'src>,
        body: Vec<Stmt<'src>>,
        else_body: Vec<Stmt<'src>>,
    },
    /// A bare expression statement.
    Expr(Expr<'src>),
    /// `pass`
    Pass,
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `import module [as alias]`
    Import {
        module: &'src str,
        alias: Option<&'src str>,
    },
    /// `from module import name [as alias]`
    FromImport {
        module: &'src str,
        name: &'src str,
        alias: Option<&'src str>,
    },
    /// A parse error: the source slice covers the bad input, message describes it.
    Error(&'src str, &'static str),
}

// ---------------------------------------------------------------------------
// Top-level items
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Item<'src> {
    pub src: &'src str,
    pub kind: ItemKind<'src>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind<'src> {
    FunctionDef(FunctionDef<'src>),
    StructDef(StructDef<'src>),
    ImplBlock(ImplBlock<'src>),
    ActorDef(ActorDef<'src>),
    /// A top-level statement (assignment, import, etc.).
    Stmt(Stmt<'src>),
    /// A parse error at item level.
    Error(&'src str, &'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<'src> {
    pub src: &'src str,
    /// The `def` or `async` keyword slice.
    pub keyword: &'src str,
    pub is_async: bool,
    pub name: &'src str,
    pub params: Vec<Param<'src>>,
    pub return_type: Option<TypeExpr<'src>>,
    pub body: Vec<Stmt<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField<'src> {
    pub src: &'src str,
    pub name: &'src str,
    pub annotation: TypeExpr<'src>,
    pub default: Option<Expr<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef<'src> {
    pub src: &'src str,
    pub name: &'src str,
    pub fields: Vec<StructField<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock<'src> {
    pub src: &'src str,
    pub type_name: &'src str,
    pub methods: Vec<FunctionDef<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActorDef<'src> {
    pub src: &'src str,
    pub name: &'src str,
    pub fields: Vec<StructField<'src>>,
    pub methods: Vec<FunctionDef<'src>>,
}

// ---------------------------------------------------------------------------
// Module (top-level)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Module<'src> {
    pub src: &'src str,
    pub items: Vec<Item<'src>>,
}
