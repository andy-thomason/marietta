//! The Marietta compiler library.
//!
//! This crate exposes all compiler passes and a high-level [`pipeline`] module
//! for driving end-to-end compilation from source text.
//!
//! # Passes (in order)
//!
//! 1. [`lexer`]          — tokenise Marietta source text
//! 2. [`parser`]         — produce an [`ast::Module`]
//! 3. [`resolve`]        — resolve names, produce a [`resolve::ResolutionMap`]
//! 4. [`types`]          — Hindley-Milner type inference
//! 5. [`async_transform`]— rewrite async fns into state-machine structs
//! 6. [`actor`]          — analyse actor declarations
//! 7. [`ir`]             — lower typed AST to three-address IR
//! 8. [`codegen`]        — Cranelift-based code generation (JIT or object file)
//!
//! The [`pipeline`] module wires these passes together into the three top-level
//! driver functions: [`pipeline::check`], [`pipeline::build`], and
//! [`pipeline::run`].

pub mod actor;
pub mod ast;
pub mod async_transform;
pub mod codegen;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod pipeline;
pub mod resolve;
pub mod types;
