/// Async / coroutine transform for the Marietta compiler.
///
/// This pass analyses every `async def` in the module and produces an
/// [`AsyncMachine`] description for each one.  Code generation (step 9) uses
/// these descriptions to emit state-machine structs and `step()` methods.
///
/// # Algorithm
///
/// An `async def` body is split at each `await` site into a sequence of
/// *resume states*:
///
/// ```text
/// state 0: stmts before first await  → Terminator::Await(future_expr)
/// state 1: stmts between first/second await → Terminator::Await(…)
/// …
/// state N: remaining stmts → Terminator::Return / Terminator::FallThrough
/// ```
///
/// All parameters and locally-declared variables are promoted to fields on
/// the state-machine struct so their values survive across suspension points.
///
/// # Limitations (v0)
///
/// * `await` must be the sole right-hand side of an assignment or appear as a
///   standalone expression statement.  `await` nested deeper in a larger
///   expression is flagged with a diagnostic and left in place.
/// * `await` inside `if`/`while`/`for` bodies emits a diagnostic; the suspend
///   is still inserted but branch-level splitting is deferred to a later pass.

use crate::ast::*;

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// One field on the generated state-machine struct.
#[derive(Debug, Clone, PartialEq)]
pub struct MachineField<'src> {
    /// The variable name (matches the source identifier).
    pub name: &'src str,
    /// Type annotation, if present in the source.
    pub annotation: Option<TypeExpr<'src>>,
}

/// How a [`ResumeState`] ends.
#[derive(Debug, Clone, PartialEq)]
pub enum StateTerminator<'src> {
    /// Suspend: drive the future produced by `expr`; on completion bind the
    /// resolved value to `bind_name` (if the await appeared in an assignment)
    /// and continue in the next state.
    Await {
        /// The expression being awaited.
        expr: Expr<'src>,
        /// Variable to receive the resolved value when resuming, if any.
        bind_name: Option<&'src str>,
    },
    /// The function returns `value` (possibly `None`).
    Return(Option<Expr<'src>>),
    /// Execution fell off the end of the function body — implicit `return None`.
    FallThrough,
}

/// One resume state within an async state machine.
///
/// The state machine dispatches on `__state` and executes `stmts`, then acts
/// on `terminator` (suspend or return).
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeState<'src> {
    /// Statements to execute upon entering this state.
    pub stmts: Vec<Stmt<'src>>,
    /// What happens at the end of this state.
    pub terminator: StateTerminator<'src>,
}

/// Complete description of the state machine to be generated from one
/// `async def`.
///
/// Code generation walks `states` in order and emits a `match __state { … }`
/// dispatch function (`step`).
#[derive(Debug, Clone, PartialEq)]
pub struct AsyncMachine<'src> {
    /// Source slice of the original `async def` keyword (for diagnostics).
    pub src: &'src str,
    /// Name of the original function, e.g. `"fetch_data"`.
    pub fn_name: &'src str,
    /// Generated struct name, e.g. `"FetchDataMachine"`.
    pub struct_name: String,
    /// Fields that persist across suspension points: params + all locals.
    pub fields: Vec<MachineField<'src>>,
    /// Return type annotation of the original function, if present.
    pub return_type: Option<TypeExpr<'src>>,
    /// Resume states in order; state 0 is always the entry point.
    pub states: Vec<ResumeState<'src>>,
}

/// A diagnostic from the async transform pass.
#[derive(Debug, Clone, PartialEq)]
pub struct AsyncDiagnostic<'src> {
    pub src: &'src str,
    pub message: &'static str,
}

/// Result returned by [`transform`].
pub struct AsyncTransformResult<'src> {
    /// One machine per `async def` found in the module (including methods).
    pub machines: Vec<AsyncMachine<'src>>,
    /// Diagnostics: unsupported await patterns, etc.
    pub diagnostics: Vec<AsyncDiagnostic<'src>>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Analyse every `async def` in `module` and return state-machine descriptions.
///
/// Non-async functions are ignored.  Methods inside `impl` blocks and
/// `actor` definitions are included.
pub fn transform<'src>(module: &Module<'src>) -> AsyncTransformResult<'src> {
    let mut machines    = Vec::new();
    let mut diagnostics = Vec::new();

    for item in &module.items {
        match &item.kind {
            ItemKind::FunctionDef(f) if f.is_async => {
                machines.push(build_machine(f, &mut diagnostics));
            }
            ItemKind::ImplBlock(ib) => {
                for method in &ib.methods {
                    if method.is_async {
                        machines.push(build_machine(method, &mut diagnostics));
                    }
                }
            }
            ItemKind::ActorDef(ad) => {
                for method in &ad.methods {
                    if method.is_async {
                        machines.push(build_machine(method, &mut diagnostics));
                    }
                }
            }
            _ => {}
        }
    }

    AsyncTransformResult { machines, diagnostics }
}

// ---------------------------------------------------------------------------
// Machine builder
// ---------------------------------------------------------------------------

fn build_machine<'src>(
    func: &FunctionDef<'src>,
    diagnostics: &mut Vec<AsyncDiagnostic<'src>>,
) -> AsyncMachine<'src> {
    // Params become the first fields (always live across the whole function).
    let mut fields: Vec<MachineField<'src>> = func.params.iter().map(|p| MachineField {
        name: p.name,
        annotation: p.annotation.clone(),
    }).collect();

    // Conservatively add all declared locals so nothing is lost across awaits.
    collect_locals(&func.body, &mut fields);

    let states = split_states(&func.body, diagnostics);

    AsyncMachine {
        src:         func.src,
        fn_name:     func.name,
        struct_name: pascal_case(func.name) + "Machine",
        fields,
        return_type: func.return_type.clone(),
        states,
    }
}

/// Convert `snake_case` (or any casing) to `PascalCase` for struct names.
fn pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None    => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Local-variable collector (conservative)
// ---------------------------------------------------------------------------

/// Push every declared variable in `stmts` (and nested blocks) onto `fields`,
/// skipping any whose name was already added.
fn collect_locals<'src>(stmts: &[Stmt<'src>], fields: &mut Vec<MachineField<'src>>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { name, annotation, .. }
            | StmtKind::LetDecl { name, annotation, .. } => {
                if !fields.iter().any(|f| f.name == *name) {
                    fields.push(MachineField { name, annotation: annotation.clone() });
                }
            }
            StmtKind::If { branches, else_body } => {
                for (_, body) in branches { collect_locals(body, fields); }
                collect_locals(else_body, fields);
            }
            StmtKind::While { body, else_body, .. }
            | StmtKind::For  { body, else_body, .. } => {
                collect_locals(body,      fields);
                collect_locals(else_body, fields);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// State splitter
// ---------------------------------------------------------------------------

/// Split `body` at every top-level `await` site, producing a list of
/// [`ResumeState`]s.  Diagnostics are emitted for unsupported patterns.
fn split_states<'src>(
    body:        &[Stmt<'src>],
    diagnostics: &mut Vec<AsyncDiagnostic<'src>>,
) -> Vec<ResumeState<'src>> {
    let mut states: Vec<ResumeState<'src>> = Vec::new();
    let mut pending: Vec<Stmt<'src>>       = Vec::new();

    for stmt in body {
        match classify_stmt(stmt) {
            StmtClass::StandaloneAwait { expr } => {
                states.push(ResumeState {
                    stmts:      std::mem::take(&mut pending),
                    terminator: StateTerminator::Await { expr, bind_name: None },
                });
            }
            StmtClass::AssignAwait { name, expr } => {
                states.push(ResumeState {
                    stmts:      std::mem::take(&mut pending),
                    terminator: StateTerminator::Await { expr, bind_name: Some(name) },
                });
            }
            StmtClass::ReturnStmt => {
                // Include the return statement then close the state.
                pending.push(stmt.clone());
                let terminator = if let StmtKind::Return(val) = &stmt.kind {
                    StateTerminator::Return(val.clone())
                } else {
                    StateTerminator::FallThrough
                };
                states.push(ResumeState { stmts: std::mem::take(&mut pending), terminator });
            }
            StmtClass::ComplexAwait => {
                // Unsupported: await nested in a larger expression.
                diagnostics.push(AsyncDiagnostic {
                    src:     stmt.src,
                    message: "await nested inside a complex expression is not yet supported; \
                              treating the statement as a suspend point",
                });
                // Best-effort: treat it as a standalone suspend with no bind.
                if let Some(expr) = extract_any_await_expr(stmt) {
                    states.push(ResumeState {
                        stmts:      std::mem::take(&mut pending),
                        terminator: StateTerminator::Await { expr, bind_name: None },
                    });
                } else {
                    pending.push(stmt.clone());
                }
            }
            StmtClass::ControlAwait => {
                // await inside if/while/for body — diagnose but still split.
                diagnostics.push(AsyncDiagnostic {
                    src:     stmt.src,
                    message: "await inside a control-flow body is not yet fully supported; \
                              splitting at this point",
                });
                pending.push(stmt.clone());
            }
            StmtClass::Plain => {
                pending.push(stmt.clone());
            }
        }
    }

    // Final state: whatever remains after the last await (or the whole body if
    // there were no awaits).
    states.push(ResumeState { stmts: pending, terminator: StateTerminator::FallThrough });

    states
}

// ---------------------------------------------------------------------------
// Statement classifier
// ---------------------------------------------------------------------------

enum StmtClass<'src> {
    /// `await expr` as a bare expression statement.
    StandaloneAwait { expr: Expr<'src> },
    /// `name = await expr` — simple assignment of an awaited value.
    AssignAwait { name: &'src str, expr: Expr<'src> },
    /// A `return` statement (no await involved).
    ReturnStmt,
    /// `await` nested inside a larger expression (not directly assignable).
    ComplexAwait,
    /// `await` appears inside a nested control-flow block.
    ControlAwait,
    /// No `await` at all.
    Plain,
}

fn classify_stmt<'src>(stmt: &Stmt<'src>) -> StmtClass<'src> {
    match &stmt.kind {
        // Standalone `await expr`
        StmtKind::Expr(Expr { kind: ExprKind::Await(inner), .. }) => {
            StmtClass::StandaloneAwait { expr: *inner.clone() }
        }

        // `name = await expr`
        StmtKind::Assign {
            target: Expr { kind: ExprKind::Name(name), .. },
            op: "=",
            value: Expr { kind: ExprKind::Await(inner), .. },
        } => StmtClass::AssignAwait { name, expr: *inner.clone() },

        // `return`
        StmtKind::Return(_) => StmtClass::ReturnStmt,

        // `if`/`while`/`for`: check bodies for nested awaits.
        StmtKind::If { branches, else_body } => {
            let in_body = branches.iter().any(|(_, b)| b.iter().any(stmt_contains_await))
                || else_body.iter().any(stmt_contains_await);
            let cond_await = branches.iter().any(|(cond, _)| expr_contains_await(cond));
            if cond_await { StmtClass::ComplexAwait }
            else if in_body { StmtClass::ControlAwait }
            else { StmtClass::Plain }
        }

        StmtKind::While { condition, body, else_body } => {
            if expr_contains_await(condition) { StmtClass::ComplexAwait }
            else if body.iter().any(stmt_contains_await)
                || else_body.iter().any(stmt_contains_await) { StmtClass::ControlAwait }
            else { StmtClass::Plain }
        }

        StmtKind::For { iter, body, else_body, .. } => {
            if expr_contains_await(iter) { StmtClass::ComplexAwait }
            else if body.iter().any(stmt_contains_await)
                || else_body.iter().any(stmt_contains_await) { StmtClass::ControlAwait }
            else { StmtClass::Plain }
        }

        // Any other statement: check for await anywhere in it.
        _ => {
            if stmt_contains_await(stmt) { StmtClass::ComplexAwait }
            else { StmtClass::Plain }
        }
    }
}

// ---------------------------------------------------------------------------
// await-presence helpers
// ---------------------------------------------------------------------------

fn stmt_contains_await(stmt: &Stmt<'_>) -> bool {
    match &stmt.kind {
        StmtKind::Expr(e)                               => expr_contains_await(e),
        StmtKind::Assign { target, value, .. }          => {
            expr_contains_await(target) || expr_contains_await(value)
        }
        StmtKind::VarDecl { value, .. }                 => {
            value.as_ref().map_or(false, expr_contains_await)
        }
        StmtKind::LetDecl { value, .. }                 => expr_contains_await(value),
        StmtKind::Return(v)                             => {
            v.as_ref().map_or(false, expr_contains_await)
        }
        StmtKind::If { branches, else_body } => {
            branches.iter().any(|(c, b)| expr_contains_await(c) || b.iter().any(stmt_contains_await))
                || else_body.iter().any(stmt_contains_await)
        }
        StmtKind::While { condition, body, else_body } => {
            expr_contains_await(condition)
                || body.iter().any(stmt_contains_await)
                || else_body.iter().any(stmt_contains_await)
        }
        StmtKind::For { iter, body, else_body, .. } => {
            expr_contains_await(iter)
                || body.iter().any(stmt_contains_await)
                || else_body.iter().any(stmt_contains_await)
        }
        _ => false,
    }
}

fn expr_contains_await(expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::Await(_) => true,
        ExprKind::BinOp { left, right, .. } => {
            expr_contains_await(left) || expr_contains_await(right)
        }
        ExprKind::UnaryOp { operand, .. }   => expr_contains_await(operand),
        ExprKind::Call { func, args, kwargs } => {
            expr_contains_await(func)
                || args.iter().any(expr_contains_await)
                || kwargs.iter().any(|(_, v)| expr_contains_await(v))
        }
        ExprKind::Index { obj, index }      => {
            expr_contains_await(obj) || expr_contains_await(index)
        }
        ExprKind::Attr { obj, .. }          => expr_contains_await(obj),
        ExprKind::IfExpr { condition, value, alt } => {
            expr_contains_await(condition)
                || expr_contains_await(value)
                || expr_contains_await(alt)
        }
        ExprKind::Lambda { body, .. }       => expr_contains_await(body),
        ExprKind::Tuple(es) | ExprKind::List(es) | ExprKind::ArrayLit(es) => {
            es.iter().any(expr_contains_await)
        }
        ExprKind::MultiSliceLit(ranges) => {
            ranges.iter().any(|(a, b)| expr_contains_await(a) || expr_contains_await(b))
        }
        _ => false,
    }
}

/// Extract the first `Await` expression found anywhere inside `stmt` (for
/// best-effort fallback when a complex await is encountered).
fn extract_any_await_expr<'src>(stmt: &Stmt<'src>) -> Option<Expr<'src>> {
    match &stmt.kind {
        StmtKind::Expr(e) | StmtKind::Return(Some(e)) => extract_from_expr(e),
        StmtKind::Assign { value, .. } => extract_from_expr(value),
        StmtKind::VarDecl { value: Some(v), .. }
        | StmtKind::LetDecl { value: v, .. } => extract_from_expr(v),
        _ => None,
    }
}

fn extract_from_expr<'src>(expr: &Expr<'src>) -> Option<Expr<'src>> {
    match &expr.kind {
        ExprKind::Await(inner) => Some(*inner.clone()),
        ExprKind::BinOp { left, right, .. } => {
            extract_from_expr(left).or_else(|| extract_from_expr(right))
        }
        ExprKind::Call { func, args, .. } => {
            extract_from_expr(func)
                .or_else(|| args.iter().find_map(extract_from_expr))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn run(src: &str) -> AsyncTransformResult<'_> {
        let pr = parser::parse(src);
        transform(&pr.module)
    }

    // ---- Non-async functions produce no machines --------------------------

    #[test]
    fn sync_fn_ignored() {
        let r = run("def add(a, b):\n    return a + b\n");
        assert!(r.machines.is_empty());
        assert!(r.diagnostics.is_empty());
    }

    // ---- Struct naming ----------------------------------------------------

    #[test]
    fn pascal_case_single_word() {
        assert_eq!(pascal_case("fetch"), "Fetch");
    }

    #[test]
    fn pascal_case_snake() {
        assert_eq!(pascal_case("fetch_data"), "FetchData");
    }

    #[test]
    fn pascal_case_leading_underscore() {
        assert_eq!(pascal_case("_helper"), "Helper");
    }

    // ---- No-await async ---------------------------------------------------

    #[test]
    fn async_no_await_one_state() {
        let r = run("async def greet():\n    return 42\n");
        assert!(r.diagnostics.is_empty());
        assert_eq!(r.machines.len(), 1);
        let m = &r.machines[0];
        assert_eq!(m.fn_name, "greet");
        assert_eq!(m.struct_name, "GreetMachine");
        // Body has one statement (return 42) → ends with FallThrough since
        // return is captured as a state with ReturnStmt terminator, then an
        // empty FallThrough state is appended. Total = 2 states.
        assert_eq!(m.states.len(), 2);
        assert!(matches!(m.states[0].terminator, StateTerminator::Return(_)));
        assert!(matches!(m.states[1].terminator, StateTerminator::FallThrough));
    }

    // ---- Single await -----------------------------------------------------

    #[test]
    fn single_await_two_states() {
        let r = run(
            "async def fetch():\n    result = await get_data()\n    return result\n",
        );
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        let m = &r.machines[0];
        // State 0: nothing before await → empty stmts, Await terminator
        // State 1: `return result`
        // State 2: FallThrough (empty tail)
        assert_eq!(m.states.len(), 3, "states: {:?}", m.states.iter().map(|s| &s.terminator).collect::<Vec<_>>());
        assert!(matches!(m.states[0].terminator, StateTerminator::Await { bind_name: Some("result"), .. }));
        assert!(matches!(m.states[1].terminator, StateTerminator::Return(_)));
    }

    // ---- Multiple awaits -------------------------------------------------

    #[test]
    fn two_awaits_three_states() {
        let r = run(
            "async def pipeline():\n    a = await step1()\n    b = await step2()\n    return a\n",
        );
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        let m = &r.machines[0];
        // State 0: before first await (empty), Await bind a
        // State 1: before second await (empty), Await bind b
        // State 2: `return a` → ReturnStmt
        // State 3: FallThrough
        assert_eq!(m.states.len(), 4);
        assert!(matches!(&m.states[0].terminator, StateTerminator::Await { bind_name: Some("a"), .. }));
        assert!(matches!(&m.states[1].terminator, StateTerminator::Await { bind_name: Some("b"), .. }));
        assert!(matches!(&m.states[2].terminator, StateTerminator::Return(_)));
    }

    // ---- Standalone await ------------------------------------------------

    #[test]
    fn standalone_await_no_bind() {
        let r = run("async def fire_and_forget():\n    await send_msg()\n");
        assert!(r.diagnostics.is_empty(), "{:?}", r.diagnostics);
        let m = &r.machines[0];
        assert!(matches!(&m.states[0].terminator, StateTerminator::Await { bind_name: None, .. }));
    }

    // ---- Params become fields --------------------------------------------

    #[test]
    fn params_are_fields() {
        let r = run("async def compute(x, y):\n    return await add(x, y)\n");
        let m = &r.machines[0];
        let names: Vec<&str> = m.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"x"), "fields: {:?}", names);
        assert!(names.contains(&"y"), "fields: {:?}", names);
    }

    // ---- Locals become fields --------------------------------------------

    #[test]
    fn locals_are_fields() {
        let r = run(
            "async def example():\n    var tmp: u32 = 0\n    tmp = await compute()\n    return tmp\n",
        );
        let m = &r.machines[0];
        let names: Vec<&str> = m.fields.iter().map(|f| f.name).collect();
        assert!(names.contains(&"tmp"), "fields: {:?}", names);
    }

    // ---- Complex await emits diagnostic ----------------------------------

    #[test]
    fn complex_await_diagnostic() {
        // `x = (await foo()) + 1` — await nested in binop
        let r = run("async def bad():\n    x = (await foo()) + 1\n");
        assert!(!r.diagnostics.is_empty(), "expected diagnostic for nested await");
    }

    // ---- Impl-block async method -----------------------------------------

    #[test]
    fn impl_async_method_found() {
        let r = run(
            "impl MyType:\n    async def load(self):\n        return await fetch()\n",
        );
        assert_eq!(r.machines.len(), 1);
        assert_eq!(r.machines[0].fn_name, "load");
    }
}
