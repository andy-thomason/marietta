mod ast;
mod lexer;
mod parser;
mod resolve;

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

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Build { file } => {
            eprintln!("build: {} (not yet implemented)", file.display());
        }
        Command::Run { file } => {
            eprintln!("run: {} (not yet implemented)", file.display());
        }
    }
}
