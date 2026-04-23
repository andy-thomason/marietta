use clap::{Parser, Subcommand};
use std::path::PathBuf;

use marietta_compiler::pipeline;

#[derive(Parser)]
#[command(name = "marietta", about = "The Marietta compiler", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check a Marietta source file for errors without generating code.
    Check {
        /// Source file to check.
        file: PathBuf,
    },
    /// Compile a Marietta source file to a native object file.
    Build {
        /// Source file to compile.
        file: PathBuf,
        /// Output object file path (defaults to <file>.o).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Compile and JIT-execute a Marietta source file.
    Run {
        /// Source file to run.
        file: PathBuf,
    },
}

fn read_source(file: &PathBuf) -> Option<String> {
    match std::fs::read_to_string(file) {
        Ok(s)  => Some(s),
        Err(e) => { eprintln!("error: cannot read {}: {e}", file.display()); None }
    }
}

fn print_result(result: &pipeline::PipelineResult) {
    for msg in &result.diagnostics {
        eprintln!("{msg}");
    }
}

fn main() {
    let cli = Cli::parse();

    let success = match cli.command {
        Command::Check { file } => {
            let Some(source) = read_source(&file) else { std::process::exit(1) };
            let result = pipeline::check(&source);
            print_result(&result);
            result.success
        }

        Command::Build { file, output } => {
            let Some(source) = read_source(&file) else { std::process::exit(1) };
            let name   = file.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            let output = output.unwrap_or_else(|| file.with_extension("o"));
            let result = pipeline::build(&source, &name, &output);
            print_result(&result);
            result.success
        }

        Command::Run { file } => {
            let Some(source) = read_source(&file) else { std::process::exit(1) };
            let result = pipeline::run(&source);
            print_result(&result);
            result.success
        }
    };

    if !success {
        std::process::exit(1);
    }
}
