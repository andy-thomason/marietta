mod ast;
mod lexer;
mod parser;
mod resolve;
mod types;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mandy", about = "The Mandy compiler", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a Mandy source file.
    Build {
        /// Source file to compile.
        file: PathBuf,
    },
    /// Compile and run a Mandy source file.
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

    !had_error
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Build { file } => {
            if !compile(&file) {
                std::process::exit(1);
            }
        }
        Command::Run { file } => {
            if !compile(&file) {
                std::process::exit(1);
            }
            eprintln!("run: codegen and execution not yet implemented");
        }
    }
}
