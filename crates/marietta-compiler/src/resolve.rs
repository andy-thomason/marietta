/// Name resolution pass.
///
/// Walks the AST produced by the parser and assigns a unique [`NodeId`] to
/// every declaration, then resolves every [`Name`] reference to the `NodeId`
/// of its binding site.
///
/// # Results
///
/// [`resolve`] returns a [`ResolveResult`] containing:
///
/// * A [`ResolutionMap`] — maps the `NodeId` of every *use* site to the
///   `NodeId` of its *declaration*.
/// * A [`Vec<ResolveDiagnostic>`] — one entry per error (undefined name,
///   use before definition, mutation of `let` binding).
///
/// # Node identity
///
/// We use the byte offset of the source slice of each AST node as its `NodeId`
/// so that no extra data structure (arena, index) is needed.  Because every
/// name reference and declaration points into the original `&'src str` the
/// offsets are stable and unique.
use std::collections::HashMap;

use crate::ast::*;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// A unique identifier for an AST node, derived from the byte offset of its
/// source slice within the original source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    fn from_src(src: &str, node_src: &str) -> Self {
        let base = src.as_ptr() as usize;
        let ptr  = node_src.as_ptr() as usize;
        NodeId(ptr.saturating_sub(base))
    }
}

// ---------------------------------------------------------------------------
// Binding kinds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum BindingKind {
    /// `var` — mutable binding.
    Var,
    /// `let` — immutable binding.
    Let,
    /// Function parameter.
    Param,
    /// Function name.
    Function,
    /// Struct name.
    Struct,
    /// Actor name.
    Actor,
    /// Trait name.
    Trait,
    /// Loop variable (`for x in …`).
    LoopVar,
    /// Implicit assignment (`x = …` with no `var`/`let`).
    Assign,
    /// Import alias or module name.
    Import,
}

#[derive(Debug, Clone)]
struct Binding<'src> {
    kind: BindingKind,
    /// The source slice of the declaration site.
    decl_src: &'src str,
    id: NodeId,
}

// ---------------------------------------------------------------------------
// Scope stack
// ---------------------------------------------------------------------------

struct Scope<'src> {
    bindings: HashMap<&'src str, Binding<'src>>,
}

impl<'src> Scope<'src> {
    fn new() -> Self {
        Scope { bindings: HashMap::new() }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ResolveDiagnostic<'src> {
    UndefinedName {
        src: &'src str,
        name: &'src str,
    },
    UseBeforeDefinition {
        src: &'src str,
        name: &'src str,
        decl_src: &'src str,
    },
    MutationOfLetBinding {
        src: &'src str,
        name: &'src str,
        decl_src: &'src str,
    },
    DuplicateDeclaration {
        src: &'src str,
        name: &'src str,
        previous_src: &'src str,
    },
    /// A trait method lacks `self` as its first parameter — not dyn-safe.
    TraitMethodNotDynSafe {
        src: &'src str,
        trait_name: &'src str,
        method_name: &'src str,
    },
    /// `impl Trait for Type` references an unknown trait name.
    ImplForUnknownTrait {
        src: &'src str,
        trait_name: &'src str,
    },
    /// `impl Trait for Type` is missing one or more required methods.
    ImplForMissingMethod {
        src: &'src str,
        trait_name: &'src str,
        method_name: &'src str,
    },
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Maps each *use-site* `NodeId` to the *declaration-site* `NodeId`.
pub type ResolutionMap = HashMap<NodeId, NodeId>;

pub struct ResolveResult<'src> {
    pub resolutions: ResolutionMap,
    pub diagnostics: Vec<ResolveDiagnostic<'src>>,
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

struct Resolver<'src> {
    source: &'src str,
    scopes: Vec<Scope<'src>>,
    resolutions: ResolutionMap,
    diagnostics: Vec<ResolveDiagnostic<'src>>,
    /// Maps trait name → list of required method names (for impl-for checking).
    trait_methods: HashMap<&'src str, Vec<&'src str>>,
}

impl<'src> Resolver<'src> {
    fn new(source: &'src str) -> Self {
        Resolver {
            source,
            scopes: vec![Scope::new()], // module-level scope
            resolutions: HashMap::new(),
            diagnostics: Vec::new(),
            trait_methods: HashMap::new(),
        }
    }

    fn node_id(&self, s: &str) -> NodeId {
        NodeId::from_src(self.source, s)
    }

    // ------------------------------------------------------------------
    // Scope helpers
    // ------------------------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Declare a name in the innermost scope.
    fn declare(&mut self, name: &'src str, decl_src: &'src str, kind: BindingKind) {
        let id = self.node_id(decl_src);
        let scope = self.scopes.last_mut().unwrap();
        if let Some(prev) = scope.bindings.get(name) {
            // Duplicate declaration in the same scope.
            self.diagnostics.push(ResolveDiagnostic::DuplicateDeclaration {
                src: decl_src,
                name,
                previous_src: prev.decl_src,
            });
        }
        scope.bindings.insert(name, Binding { kind, decl_src, id });
    }

    /// Resolve a name use — looks outward through the scope stack.
    fn resolve_name(&mut self, use_src: &'src str, name: &'src str) {
        let use_id = self.node_id(use_src);
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                self.resolutions.insert(use_id, binding.id);
                return;
            }
        }
        self.diagnostics.push(ResolveDiagnostic::UndefinedName { src: use_src, name });
    }

    /// Check that an assignment target is not a `let` binding.
    fn check_assign_target(&mut self, name: &'src str, use_src: &'src str) {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                if binding.kind == BindingKind::Let {
                    self.diagnostics.push(ResolveDiagnostic::MutationOfLetBinding {
                        src: use_src,
                        name,
                        decl_src: binding.decl_src,
                    });
                }
                return;
            }
        }
        // Not yet declared — treat as implicit declaration (like Python).
        let id = self.node_id(use_src);
        let scope = self.scopes.last_mut().unwrap();
        scope.bindings.insert(name, Binding { kind: BindingKind::Assign, decl_src: use_src, id });
    }

    // ------------------------------------------------------------------
    // Expression walker
    // ------------------------------------------------------------------

    fn walk_expr(&mut self, expr: &Expr<'src>) {
        match &expr.kind {
            ExprKind::Name(name) => self.resolve_name(expr.src, name),

            ExprKind::BinOp { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            ExprKind::UnaryOp { operand, .. } => self.walk_expr(operand),
            ExprKind::Await(inner) => self.walk_expr(inner),

            ExprKind::Call { func, args, kwargs } => {
                self.walk_expr(func);
                for a in args { self.walk_expr(a); }
                for (_, v) in kwargs { self.walk_expr(v); }
            }
            ExprKind::Index { obj, index } => {
                self.walk_expr(obj);
                self.walk_expr(index);
            }
            ExprKind::Attr { obj, .. } => self.walk_expr(obj),

            ExprKind::IfExpr { condition, value, alt } => {
                self.walk_expr(condition);
                self.walk_expr(value);
                self.walk_expr(alt);
            }
            ExprKind::Lambda { params, body } => {
                self.push_scope();
                for p in params {
                    self.declare(p.name, p.src, BindingKind::Param);
                    if let Some(default) = &p.default { self.walk_expr(default); }
                }
                self.walk_expr(body);
                self.pop_scope();
            }
            ExprKind::Tuple(elems) | ExprKind::List(elems) | ExprKind::ArrayLit(elems) => {
                for e in elems { self.walk_expr(e); }
            }
            ExprKind::MultiSliceLit(ranges) => {
                for (lo, hi) in ranges {
                    self.walk_expr(lo);
                    self.walk_expr(hi);
                }
            }

            ExprKind::StructInit { type_name, fields } => {
                // Resolve the struct type name
                self.resolve_name(expr.src, type_name);
                // Walk each field initialization expression
                for (_, value) in fields {
                    self.walk_expr(value);
                }
            }

            // Literals and errors carry no names.
            ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_)
            | ExprKind::StringLiteral(_) | ExprKind::BoolLiteral(_)
            | ExprKind::NoneLiteral | ExprKind::Error(_) => {}
        }
    }

    // ------------------------------------------------------------------
    // Assign target walker (extracts names being bound)
    // ------------------------------------------------------------------

    fn walk_assign_target(&mut self, target: &Expr<'src>, is_augmented: bool) {
        match &target.kind {
            ExprKind::Name(name) => {
                if is_augmented {
                    // Augmented assign requires the name to already exist.
                    self.resolve_name(target.src, name);
                    self.check_assign_target(name, target.src);
                } else {
                    self.check_assign_target(name, target.src);
                }
            }
            ExprKind::Tuple(elems) | ExprKind::List(elems) | ExprKind::ArrayLit(elems) => {
                for e in elems { self.walk_assign_target(e, is_augmented); }
            }
            ExprKind::MultiSliceLit(ranges) => {
                for (lo, hi) in ranges {
                    self.walk_expr(lo);
                    self.walk_expr(hi);
                }
            }
            ExprKind::Attr { obj, .. } => self.walk_expr(obj),
            ExprKind::Index { obj, index } => {
                self.walk_expr(obj);
                self.walk_expr(index);
            }
            _ => self.walk_expr(target),
        }
    }

    // ------------------------------------------------------------------
    // Statement walker
    // ------------------------------------------------------------------

    fn walk_stmt(&mut self, stmt: &Stmt<'src>) {
        match &stmt.kind {
            StmtKind::Assign { op, target, value } => {
                self.walk_expr(value);
                let is_aug = *op != "=";
                self.walk_assign_target(target, is_aug);
            }
            StmtKind::VarDecl { name, annotation: _, value } => {
                if let Some(v) = value { self.walk_expr(v); }
                self.declare(name, stmt.src, BindingKind::Var);
            }
            StmtKind::LetDecl { name, annotation: _, value } => {
                self.walk_expr(value);
                self.declare(name, stmt.src, BindingKind::Let);
            }
            StmtKind::Return(value) => {
                if let Some(v) = value { self.walk_expr(v); }
            }
            StmtKind::If { branches, else_body } => {
                for (cond, body) in branches {
                    self.walk_expr(cond);
                    self.push_scope();
                    for s in body { self.walk_stmt(s); }
                    self.pop_scope();
                }
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }
            StmtKind::While { condition, body, else_body } => {
                self.walk_expr(condition);
                self.push_scope();
                for s in body { self.walk_stmt(s); }
                self.pop_scope();
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }
            StmtKind::For { target, iter, body, else_body } => {
                self.walk_expr(iter);
                self.push_scope();
                self.walk_assign_target(target, false);
                for s in body { self.walk_stmt(s); }
                self.pop_scope();
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }
            StmtKind::Expr(expr) => self.walk_expr(expr),
            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Import { module, alias } => {
                let binding_name = alias.unwrap_or(module);
                self.declare(binding_name, stmt.src, BindingKind::Import);
            }
            StmtKind::FromImport { name, alias, .. } => {
                let binding_name = alias.unwrap_or(name);
                self.declare(binding_name, stmt.src, BindingKind::Import);
            }
            StmtKind::Error(_, _) => {}
        }
    }

    // ------------------------------------------------------------------
    // Function definition
    // ------------------------------------------------------------------

    fn walk_function(&mut self, f: &FunctionDef<'src>) {
        self.push_scope();
        for p in &f.params {
            // Evaluate defaults in the *outer* scope (Python semantics).
            if let Some(default) = &p.default { self.walk_expr(default); }
            self.declare(p.name, p.src, BindingKind::Param);
        }
        for s in &f.body { self.walk_stmt(s); }
        self.pop_scope();
    }

    // ------------------------------------------------------------------
    // Top-level items
    // ------------------------------------------------------------------

    fn walk_item(&mut self, item: &Item<'src>) {
        match &item.kind {
            ItemKind::FunctionDef(f) => {
                self.declare(f.name, f.src, BindingKind::Function);
                self.walk_function(f);
            }
            ItemKind::StructDef(s) => {
                self.declare(s.name, s.src, BindingKind::Struct);
                // Field defaults are evaluated at struct-level scope.
                for field in &s.fields {
                    if let Some(default) = &field.default { self.walk_expr(default); }
                }
            }
            ItemKind::ImplBlock(i) => {
                // `impl Foo:` — Foo must already be declared.
                self.resolve_name(i.src, i.type_name);
                for method in &i.methods {
                    self.walk_function(method);
                }
            }
            ItemKind::TraitDef(t) => {
                // Declaration was already hoisted; only run dyn-safety checks here.
                for method in &t.methods {
                    let is_dyn_safe = method.params.first()
                        .map(|p| p.name == "self")
                        .unwrap_or(false);
                    if !is_dyn_safe {
                        self.diagnostics.push(ResolveDiagnostic::TraitMethodNotDynSafe {
                            src: method.src,
                            trait_name: t.name,
                            method_name: method.name,
                        });
                    }
                }
            }
            ItemKind::ImplFor(i) => {
                // Resolve both the trait and the type.
                self.resolve_name(i.src, i.trait_name);
                self.resolve_name(i.src, i.type_name);
                // Check all trait methods are implemented.
                if let Some(required) = self.trait_methods.get(i.trait_name).cloned() {
                    for req in &required {
                        if !i.methods.iter().any(|m| m.name == *req) {
                            self.diagnostics.push(ResolveDiagnostic::ImplForMissingMethod {
                                src: i.src,
                                trait_name: i.trait_name,
                                method_name: req,
                            });
                        }
                    }
                } else {
                    self.diagnostics.push(ResolveDiagnostic::ImplForUnknownTrait {
                        src: i.src,
                        trait_name: i.trait_name,
                    });
                }
                for method in &i.methods {
                    self.walk_function(method);
                }
            }
            ItemKind::ActorDef(a) => {
                self.declare(a.name, a.src, BindingKind::Actor);
                for field in &a.fields {
                    if let Some(default) = &field.default { self.walk_expr(default); }
                }
                for method in &a.methods {
                    self.walk_function(method);
                }
            }
            ItemKind::Stmt(stmt) => self.walk_stmt(stmt),
            ItemKind::Error(_, _) => {}
        }
    }

    fn walk_module(&mut self, module: &Module<'src>) {
        // First pass: hoist all top-level function/struct/actor/trait names so that
        // forward references within the module work.
        for item in &module.items {
            match &item.kind {
                ItemKind::FunctionDef(f) => {
                    self.declare(f.name, f.src, BindingKind::Function);
                }
                ItemKind::StructDef(s) => {
                    self.declare(s.name, s.src, BindingKind::Struct);
                }
                ItemKind::ActorDef(a) => {
                    self.declare(a.name, a.src, BindingKind::Actor);
                }
                ItemKind::TraitDef(t) => {
                    self.declare(t.name, t.src, BindingKind::Trait);
                    // Pre-populate trait_methods so ImplFor can check completeness
                    // even when the impl comes before the trait in source order.
                    let names: Vec<&'src str> = t.methods.iter().map(|m| m.name).collect();
                    self.trait_methods.insert(t.name, names);
                }
                _ => {}
            }
        }
        // Second pass: resolve everything.
        for item in &module.items {
            match &item.kind {
                // Already declared in hoist pass; skip re-declaration.
                ItemKind::FunctionDef(f) => self.walk_function(f),
                ItemKind::StructDef(s) => {
                    for field in &s.fields {
                        if let Some(default) = &field.default { self.walk_expr(default); }
                    }
                }
                ItemKind::ActorDef(a) => {
                    for field in &a.fields {
                        if let Some(default) = &field.default { self.walk_expr(default); }
                    }
                    for method in &a.methods { self.walk_function(method); }
                }
                // TraitDef already declared in hoist pass; only do dyn-safety checks.
                ItemKind::TraitDef(_) => self.walk_item(item),
                _ => self.walk_item(item),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn resolve<'src>(source: &'src str, module: &Module<'src>) -> ResolveResult<'src> {
    let mut resolver = Resolver::new(source);
    resolver.walk_module(module);
    ResolveResult {
        resolutions: resolver.resolutions,
        diagnostics: resolver.diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn run(src: &str) -> ResolveResult<'_> {
        let pr = parser::parse(src);
        let result = resolve(src, &pr.module);
        result
    }

    fn is_clean(src: &str) -> bool {
        run(src).diagnostics.is_empty()
    }

    // ---- No errors expected ------------------------------------------------

    #[test]
    fn simple_assignment() {
        assert!(is_clean("x = 1\n"));
    }

    #[test]
    fn var_decl_and_use() {
        assert!(is_clean("var x: u8 = 1\ny = x\n"));
    }

    #[test]
    fn let_decl_and_use() {
        assert!(is_clean("let x: u8 = 1\ny = x\n"));
    }

    #[test]
    fn function_def_and_call() {
        assert!(is_clean("def add(a, b):\n    return a + b\nresult = add(1, 2)\n"));
    }

    #[test]
    fn nested_scopes() {
        // Functions close over outer-scope variables.
        assert!(is_clean(
            "x = 1\ndef get_x():\n    return x\n"
        ));
    }

    #[test]
    fn for_loop_var() {
        assert!(is_clean("items = 1\nfor i in items:\n    x = i\n"));
    }

    #[test]
    fn if_else() {
        assert!(is_clean("x = 1\nif x:\n    y = 2\nelse:\n    y = 3\n"));
    }

    #[test]
    fn import_use() {
        assert!(is_clean("import math\nx = math\n"));
    }

    #[test]
    fn forward_reference_to_function() {
        // Top-level functions are hoisted so a call before the def is valid.
        assert!(is_clean("result = add(1, 2)\ndef add(a, b):\n    return a + b\n"));
    }

    #[test]
    fn struct_def() {
        assert!(is_clean("struct Point:\n    x: f32\n    y: f32\n"));
    }

    #[test]
    fn impl_block_after_struct() {
        assert!(is_clean(
            "struct Point:\n    x: f32\nimpl Point:\n    def norm(self) -> f32:\n        return self.x\n"
        ));
    }

    #[test]
    fn actor_def() {
        assert!(is_clean(
            "actor Counter:\n    var count: u64 = 0\n    def increment(self):\n        self.count += 1\n"
        ));
    }

    #[test]
    fn augmented_assign() {
        assert!(is_clean("x = 0\nx += 1\n"));
    }

    #[test]
    fn lambda() {
        assert!(is_clean("f = lambda x: x + 1\n"));
    }

    #[test]
    fn lambda_multi_param() {
        assert!(is_clean("f = lambda x, y: x + y\n"));
    }

    #[test]
    fn resolution_map_populated() {
        let src = "var x: u8 = 1\ny = x\n";
        let result = run(src);
        assert!(!result.resolutions.is_empty());
    }

    // ---- Errors expected ---------------------------------------------------

    #[test]
    fn undefined_name() {
        let diags = run("y = x\n").diagnostics;
        assert!(diags.iter().any(|d| matches!(d, ResolveDiagnostic::UndefinedName { name: "x", .. })));
    }

    #[test]
    fn mutation_of_let() {
        let diags = run("let x: u8 = 1\nx = 2\n").diagnostics;
        assert!(
            diags.iter().any(|d| matches!(d, ResolveDiagnostic::MutationOfLetBinding { name: "x", .. })),
            "expected MutationOfLetBinding, got: {:?}", diags
        );
    }

    #[test]
    fn undefined_in_function() {
        let diags = run("def f():\n    return z\n").diagnostics;
        assert!(diags.iter().any(|d| matches!(d, ResolveDiagnostic::UndefinedName { name: "z", .. })));
    }

    #[test]
    fn impl_on_unknown_type() {
        let diags = run("impl Ghost:\n    def f(self):\n        pass\n").diagnostics;
        assert!(diags.iter().any(|d| matches!(d, ResolveDiagnostic::UndefinedName { name: "Ghost", .. })));
    }

    #[test]
    fn multiple_errors_reported() {
        // Both `a` and `b` are undefined; both should be reported.
        let diags = run("x = a + b\n").diagnostics;
        assert_eq!(diags.len(), 2, "got: {:?}", diags);
    }

    #[test]
    fn no_panic_on_error_nodes() {
        // Parser errors should not cause the resolver to panic.
        let _ = run("x = \n");
    }

    // ---- Source slice identity --------------------------------------------

    #[test]
    fn diagnostic_src_in_source() {
        let src = "y = undefined_name\n";
        let result = run(src);
        for d in &result.diagnostics {
            let diag_ptr = match d {
                ResolveDiagnostic::UndefinedName { src: s, .. } => s.as_ptr() as usize,
                _ => continue,
            };
            let base = src.as_ptr() as usize;
            assert!(diag_ptr >= base && diag_ptr < base + src.len());
        }
    }

    // ---- Trait & dyn tests ------------------------------------------------

    #[test]
    fn trait_def_no_error() {
        // A trait whose only method has `self` is dyn-safe.
        assert!(is_clean("trait Drawable:\n    def draw(self) -> None:\n"));
    }

    #[test]
    fn trait_method_not_dyn_safe() {
        // A method without `self` violates dyn-safety.
        let diags = run("trait Foo:\n    def static_method() -> None:\n").diagnostics;
        assert!(
            diags.iter().any(|d| matches!(d,
                ResolveDiagnostic::TraitMethodNotDynSafe { trait_name: "Foo", method_name: "static_method", .. }
            )),
            "expected TraitMethodNotDynSafe, got: {:?}", diags
        );
    }

    #[test]
    fn impl_for_complete() {
        // All methods implemented — no diagnostics.
        assert!(is_clean(
            "struct Dog:\n    x: u8\n\
             trait Animal:\n    def speak(self) -> None:\n\
             impl Animal for Dog:\n    def speak(self) -> None:\n        pass\n"
        ));
    }

    #[test]
    fn impl_for_missing_method() {
        let diags = run(
            "struct Cat:\n    x: u8\n\
             trait Animal:\n    def speak(self) -> None:\n\
             impl Animal for Cat:\n    def other(self) -> None:\n        pass\n"
        ).diagnostics;
        assert!(
            diags.iter().any(|d| matches!(d,
                ResolveDiagnostic::ImplForMissingMethod { trait_name: "Animal", method_name: "speak", .. }
            )),
            "expected ImplForMissingMethod, got: {:?}", diags
        );
    }

    #[test]
    fn impl_for_unknown_trait() {
        let diags = run(
            "struct Dog:\n    x: u8\n\
             impl NoSuchTrait for Dog:\n    def speak(self) -> None:\n        pass\n"
        ).diagnostics;
        assert!(
            diags.iter().any(|d| matches!(d,
                ResolveDiagnostic::ImplForUnknownTrait { trait_name: "NoSuchTrait", .. }
            )),
            "expected ImplForUnknownTrait, got: {:?}", diags
        );
    }
}

