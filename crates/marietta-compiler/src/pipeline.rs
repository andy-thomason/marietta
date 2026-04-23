//! High-level compilation pipeline for the Marietta compiler.
//!
//! Three entry points mirror the three CLI subcommands:
//!
//! | Function          | What it does                                      |
//! |-------------------|---------------------------------------------------|
//! | [`check`]         | Parse → resolve → type-check → IR (no codegen)   |
//! | [`build`]         | Full pipeline → native object file                |
//! | [`run`]           | Full pipeline → JIT-execute `main()`              |

use std::path::Path;

use crate::{actor, async_transform, codegen, ir, parser, resolve, types};

// ---------------------------------------------------------------------------
// Public result type
// ---------------------------------------------------------------------------

/// The outcome of a compilation pipeline run.
#[derive(Debug, Default)]
pub struct PipelineResult {
    /// `true` when there were no hard errors (warnings are fine).
    pub success: bool,
    /// Human-readable messages in the order they were emitted.
    pub diagnostics: Vec<String>,
}

impl PipelineResult {
    fn ok() -> Self { Self { success: true,  diagnostics: Vec::new() } }
    fn fail() -> Self { Self { success: false, diagnostics: Vec::new() } }
    fn push(&mut self, msg: impl Into<String>) { self.diagnostics.push(msg.into()); }
}

// ---------------------------------------------------------------------------
// Shared front-end (parse → resolve → type-check)
// ---------------------------------------------------------------------------

struct FrontEnd<'src> {
    source:        &'src str,
    parse_result:  parser::ParseResult<'src>,
    resolve_result: resolve::ResolveResult<'src>,
    infer_result:  types::InferResult<'src>,
    had_error:     bool,
    diagnostics:   Vec<String>,
}

impl<'src> FrontEnd<'src> {
    fn run(source: &'src str) -> Self {
        let mut diagnostics = Vec::new();
        let mut had_error = false;

        let parse_result = parser::parse(source);
        for d in &parse_result.diagnostics {
            diagnostics.push(format!("parse error: {}", d.message));
            had_error = true;
        }

        let resolve_result = resolve::resolve(source, &parse_result.module);
        for d in &resolve_result.diagnostics {
            diagnostics.push(format!("name error: {d:?}"));
            had_error = true;
        }

        let infer_result = types::infer(source, &parse_result.module, &resolve_result.resolutions);
        for d in &infer_result.diagnostics {
            diagnostics.push(format!("type error: {d:?}"));
            had_error = true;
        }

        FrontEnd { source, parse_result, resolve_result, infer_result, had_error, diagnostics }
    }
}

// ---------------------------------------------------------------------------
// check — analysis only, no code generation
// ---------------------------------------------------------------------------

/// Run all analysis passes without generating code.
///
/// Suitable for `marietta check`: reports parse, name-resolution, and type
/// errors as quickly as possible.
pub fn check(source: &str) -> PipelineResult {
    let mut fe = FrontEnd::run(source);
    let mut res = if fe.had_error { PipelineResult::fail() } else { PipelineResult::ok() };
    res.diagnostics.append(&mut fe.diagnostics);

    let async_result = async_transform::transform(&fe.parse_result.module);
    for d in &async_result.diagnostics {
        res.push(format!("async warning: {}", d.message));
    }
    if !async_result.machines.is_empty() {
        res.push(format!("info: {} async state machine(s) found", async_result.machines.len()));
    }

    let actor_result = actor::analyse(&fe.parse_result.module);
    for d in &actor_result.diagnostics {
        res.push(format!("actor warning: {}", d.message));
    }
    if !actor_result.analyses.is_empty() {
        res.push(format!("info: {} actor(s) found", actor_result.analyses.len()));
    }

    let ir_module = ir::lower(fe.source, &fe.parse_result.module, &mut fe.infer_result);
    for d in &ir_module.diagnostics {
        res.push(format!("ir warning: {}", d.message));
    }
    if !ir_module.functions.is_empty() {
        res.push(format!("info: {} function(s) lowered to IR", ir_module.functions.len()));
    }

    res
}

// ---------------------------------------------------------------------------
// build — compile to native object file
// ---------------------------------------------------------------------------

/// Compile `source` to a native object file at `output`.
///
/// `name` is used as the module name embedded in the object file (typically
/// the stem of the source file name).
pub fn build(source: &str, name: &str, output: &Path) -> PipelineResult {
    let mut fe = FrontEnd::run(source);
    let mut res = if fe.had_error { PipelineResult::fail() } else { PipelineResult::ok() };
    res.diagnostics.append(&mut fe.diagnostics);
    if fe.had_error { return res; }

    let ir_module = ir::lower(fe.source, &fe.parse_result.module, &mut fe.infer_result);
    for d in &ir_module.diagnostics {
        res.push(format!("ir warning: {}", d.message));
    }

    let diags = codegen::codegen_object(&ir_module, name, output);
    for d in &diags {
        res.push(format!("codegen: {}", d.message));
    }

    if diags.iter().any(|d| d.message.contains("failed")) {
        res.success = false;
    } else {
        res.push(format!("info: object written to {}", output.display()));
    }

    res
}

// ---------------------------------------------------------------------------
// run — JIT-compile and execute
// ---------------------------------------------------------------------------

/// JIT-compile `source` and execute its `main()` function, if present.
pub fn run(source: &str) -> PipelineResult {
    let mut fe = FrontEnd::run(source);
    let mut res = if fe.had_error { PipelineResult::fail() } else { PipelineResult::ok() };
    res.diagnostics.append(&mut fe.diagnostics);
    if fe.had_error { return res; }

    let ir_module = ir::lower(fe.source, &fe.parse_result.module, &mut fe.infer_result);
    for d in &ir_module.diagnostics {
        res.push(format!("ir warning: {}", d.message));
    }

    let artifact = codegen::codegen_jit(&ir_module);
    for d in &artifact.diagnostics {
        res.push(format!("codegen: {}", d.message));
    }

    // Execute main() if present.
    // SAFETY: we cast the raw pointer to the expected C ABI signature.  The
    // JitArtifact keeps the code alive for the duration of this scope.
    if let Some(ptr) = unsafe { artifact.fn_ptr("main") } {
        res.push("info: executing main()".to_string());
        let f: extern "C" fn() = unsafe { std::mem::transmute(ptr) };
        unsafe { f() };
    } else {
        res.push("info: no main() function — compilation succeeded".to_string());
    }

    res
}
