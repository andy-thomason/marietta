mod actor;
mod ast;
mod async_transform;
mod codegen;
mod ir;
mod lexer;
mod parser;
mod resolve;
mod types;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "marietta", about = "The Marietta compiler", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a Marietta source file.
    Build {
        /// Source file to compile.
        file: PathBuf,
    },
    /// Compile and run a Marietta source file.
    Run {
        /// Source file to run.
        file: PathBuf,
    },
}

fn compile(file: &PathBuf) -> bool {
    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", file.display(), e);
            return false;
        }
    };

    let parse_result = parser::parse(&source);
    let mut had_error = false;

    for diag in &parse_result.diagnostics {
        eprintln!("parse error: {}", diag.message);
        had_error = true;
    }

    let resolve_result = resolve::resolve(&source, &parse_result.module);

    for diag in &resolve_result.diagnostics {
        eprintln!("name error: {:?}", diag);
        had_error = true;
    }

    let mut infer_result = types::infer(&source, &parse_result.module, &resolve_result.resolutions);

    for diag in &infer_result.diagnostics {
        eprintln!("type error: {:?}", diag);
        had_error = true;
    }

    let async_result = async_transform::transform(&parse_result.module);

    for diag in &async_result.diagnostics {
        eprintln!("async warning: {}", diag.message);
        // Async transform diagnostics are warnings, not hard errors.
    }

    if !async_result.machines.is_empty() {
        eprintln!("info: {} async state machine(s) found", async_result.machines.len());
    }

    let actor_result = actor::analyse(&parse_result.module);

    for diag in &actor_result.diagnostics {
        eprintln!("actor warning: {}", diag.message);
        // Actor diagnostics are warnings, not hard errors.
    }

    if !actor_result.analyses.is_empty() {
        eprintln!("info: {} actor(s) found", actor_result.analyses.len());
    }

    let ir_module = ir::lower(&source, &parse_result.module, &mut infer_result);

    for diag in &ir_module.diagnostics {
        eprintln!("ir warning: {}", diag.message);
    }

    if !ir_module.functions.is_empty() {
        eprintln!("info: {} function(s) lowered to IR", ir_module.functions.len());
    }

    !had_error
}

fn build(file: &PathBuf) -> bool {
    let source = match std::fs::read_to_string(file) {
        Ok(s)  => s,
        Err(e) => { eprintln!("error: cannot read {}: {}", file.display(), e); return false; }
    };
    let parse_result  = parser::parse(&source);
    let resolve_result = resolve::resolve(&source, &parse_result.module);
    let mut infer     = types::infer(&source, &parse_result.module, &resolve_result.resolutions);
    let ir_module     = ir::lower(&source, &parse_result.module, &mut infer);

    let stem   = file.file_stem().unwrap_or_default().to_string_lossy();
    let output = file.with_extension("o");
    let diags  = codegen::codegen_object(&ir_module, &stem, &output);
    for d in &diags {
        eprintln!("codegen: {}", d.message);
    }
    let fatal = diags.iter().any(|d| d.message.contains("failed"));
    if !fatal {
        eprintln!("info: object written to {}", output.display());
    }
    !fatal
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Build { file } => {
            if !build(&file) {
                std::process::exit(1);
            }
        }
        Command::Run { file } => {
            let source = match std::fs::read_to_string(&file) {
                Ok(s)  => s,
                Err(e) => { eprintln!("error: cannot read {}: {}", file.display(), e); std::process::exit(1); }
            };
            let parse_result   = parser::parse(&source);
            let resolve_result = resolve::resolve(&source, &parse_result.module);
            let mut infer      = types::infer(&source, &parse_result.module, &resolve_result.resolutions);
            let ir_module      = ir::lower(&source, &parse_result.module, &mut infer);

            let artifact = codegen::codegen_jit(&ir_module);
            for d in &artifact.diagnostics {
                eprintln!("codegen: {}", d.message);
            }

            // If a `main` function exists, execute it.
            if let Some(ptr) = unsafe { artifact.fn_ptr("main") } {
                eprintln!("info: executing main()");
                let f: extern "C" fn() = unsafe { std::mem::transmute(ptr) };
                unsafe { f() };
            } else {
                eprintln!("info: no main() function found — compilation succeeded");
            }
        }
    }
}
