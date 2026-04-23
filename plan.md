# The Marietta programming language

Marietta is python-like but compiled to concrete values. It uses Python-like syntax
but leans more on Rust for implementation. The objective is to make actor models
with channels and RPC calls very easy to implement.

Every function can be async, ie. it returns a struct containing the variables
as well as an entrypoint for the function.

Steps.

* Construct a Python lexer. This takes a source string and returns tokens that reference
that string. Each token will contain a single string reference which will be used for
source correspondence and reporting errors. The Token enum should be minimal, eg. Punctuation,
Identifier, Keyword all with strings such as Keyword("if"). Errors have an Error token.
Indentation needs a token eg. Indentation("    ") and Indentation(""). No tabs are allowed.

* Construct a Python parser that generates AST. The AST elements contain string references
to the original string for source correspondence. Use a Pratt parser. Try to handle errors
gracefully, reporting errors while parsing but always accepting the parsed result. This
means that errors at the end of the program are still reported.

* Variable types are like Mojo. Assignment to a variable fixes the type but types may be specified.
We need a pass to infer the types of variables.

eg.

```
def variables():
    a = 1
    var x: u8 = 10
    var sum: u8 = x + x
    let name: String = "Marietta"
    # name = "Python" # Error: Cannot mutate let value
    print(sum)
```

Integer types are as in Rust but any number of bits (u1-u2048) (i1-i2048)

Floating point types use bit size and mantissa size like f16_8 for bf16 or just f32 for ieee mantissa sizes.

Reference types are as in Rust with slices allowed. For sub-byte types, slices use bits instead of bytes for
the pointer. Pointers are also allowed as in Rust.

Multidimensional slices are also possible. Given a tensor, we should be able to slice as &[10..20, 30..40]
In this case we need strides and lengths for all dimensions.

Multiplying slices with %*% applies tensor multiply rules yielding an array. Binops/Unops slices works element-wise as in R.

* Structs are similar to mojo, but we have separate impl blocks as in rust for methods.

* Write some examples of Marietta for testing. A 256 bit fibanocci, a matrix multiply. We may revise the syntax down the line.

---

## Detailed Implementation Steps

### 1. Lexer (`src/lexer.rs`)

- Define a `Token<'src>` enum with variants:
  - `Keyword(&'src str)` — `def`, `let`, `var`, `if`, `else`, `elif`, `for`, `while`,
    `return`, `import`, `struct`, `impl`, `async`, `await`, `actor`, `channel`, `rpc`
  - `Identifier(&'src str)`
  - `IntLiteral(&'src str)` — raw text, parsed later during type checking
  - `FloatLiteral(&'src str)`
  - `StringLiteral(&'src str)` — includes delimiters for span accuracy
  - `Punctuation(&'src str)` — single or double-character operators and brackets
  - `Indent(&'src str)` — Length of string is the column depth.
  - `Dedent` — emitted when indentation decreases; may emit multiple
  - `Newline`
  - `Comment(&'src str)`
  - `Error(&'src str)` — unrecognised character, tab, or malformed token
  - `Eof`
- We do not need a Span as the string reference gives the source correspondence.
- The lexer is a struct that holds `&'src str` and a cursor; it implements `Iterator<Item = Token<'src>>`.
- Track indent stack as `Vec<usize>`; emit `Indent`/`Dedent` on line changes.
- Reject hard tabs with an `Error` token and a message.
- Unit-test each token variant with `#[cfg(test)]` inline tests.

### 2. AST (`src/ast.rs`)

- We do not need a Span as the string reference gives the source correspondence Make sure that every node contains the full text reference of the production in covers as well as the token that differentiates it. eg. BinOp("+", "1 + 1") The strings in the tokens are guaranteed to come from the same str.
- Define expression nodes: `Literal`, `Name`, `BinOp`, `UnaryOp`, `Call`, `Index`,
  `Attr`, `Await`, `If` (ternary), `Lambda`.
- Define statement nodes: `Assign`, `VarDecl`, `LetDecl`, `Return`, `If`, `While`,
  `For`, `Expr`, `Pass`, `Break`, `Continue`, `Import`.
- Define top-level items: `FunctionDef`, `StructDef`, `ImplBlock`, `ActorDef`.
- Each node stores its child nodes by value (use `Box<Expr>` for
  recursive types).
- Keep a parallel `ErrorNode` that wraps a source str and message so partial trees
  can be retained after a parse error.

### 3. Parser (`src/parser.rs`)

- Implement a recursive-descent parser that consumes the token iterator.
- Use a Pratt parser for expressions; define binding powers for all operators
  matching Python precedence.
- The parser never panics; on unexpected tokens it emits an `ErrorNode`, skips
  to a synchronisation point (next newline or `Dedent`), and continues.
- Collect all `ErrorNode`s in a `Vec<Diagnostic>` stored on the parser; return
  the (possibly partial) tree together with the diagnostics.
- Write integration tests that round-trip small Marietta snippets and check the
  resulting AST shape.

### 4. Name Resolution (`src/resolve.rs`)

- Walk the AST in a single pass maintaining a scope stack of `HashMap<&str, NodeId>`.
- Resolve each `Name` reference to the `NodeId` of its declaration.
- Report use-before-def, undefined names, and shadowing of `let` bindings as
  diagnostics.
- Produce a `ResolutionMap: HashMap<NodeId, NodeId>` for later passes.

### 5. Type Inference (`src/types.rs`)

- Define a `Type` enum: `Int { bits: u16, signed: bool }`, `Float { bits: u16, mantissa: u16 }`,
  `Bool`, `String`, `Tuple(Vec<Type>)`, `Struct(StructId)`, `Fn(Vec<Type>, Box<Type>)`,
  `Async(Box<Type>)`, `Channel(Box<Type>)`, `Unknown`, `Error`.
- Run Hindley-Milner–style unification. Start with `Unknown` for every unresolved
  expression, then propagate constraints from literals, operators, and annotations.
- Numeric literals widen to the smallest fitting type unless an annotation forces
  a specific bit width.
- Integer types are any bit width 1–2048; float types encode `bits` and `mantissa`
  (e.g. `f16_8` = bf16, `f32` = IEEE).
- Emit type-mismatch diagnostics with source spans; do not abort — insert an
  `Error` type and continue so as many errors as possible are surfaced.

### 6. Async / Coroutine Transform (`src/async_transform.rs`)

- Every `async def` is rewritten into a state-machine struct containing:
  - The local variables as fields.
  - A `step(state: &mut Self) -> Poll<ReturnType>` method.
  - An integer `__state` field tracking the current resume point.
- `await expr` becomes a yield point: save state, return `Poll::Pending`.
- This transform happens on the typed AST before code generation.
- The output is a plain (non-async) AST that code generation can handle uniformly.

### 7. Actor Model (`src/actor.rs`)

- Define the syntax for actor declarations:
  ```
  actor Counter:
      var count: u64 = 0

      def increment(self):
          self.count += 1

      def get(self) -> u64:
          return self.count
  ```
- An actor compiles to a struct plus a message enum (one variant per public method).
- The runtime spawns each actor on its own lightweight task; method calls become
  channel sends and (for non-void methods) awaited replies.
- RPC calls across actors use the same channel mechanism with a reply address.
- Expose `channel<T>` as a first-class type backed by the runtime's MPSC queue.

### 8. Intermediate Representation (`src/ir.rs`)

- Lower the typed AST to a simple three-address IR:
  - Basic blocks with a terminator (branch, return, jump).
  - SSA-form values: `%0`, `%1`, …
  - Instructions: `Add`, `Sub`, `Mul`, `Div`, `CmpEq`, `CmpLt`, `Call`,
    `Load`, `Store`, `Alloc`, `Phi`.
- Big-integer arithmetic for widths > 64 is lowered to multi-limb operations here.
- Keep span information on every instruction for debuginfo.

### 9. Code Generation (`src/codegen.rs`)

- Target: Cranelift (`cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`,
  `cranelift-module`). No LLVM dependency required.
- Each IR basic block is translated to a Cranelift `Block`; values are SSA
  `cranelift_codegen::ir::Value`s.
- Use `cranelift_jit::JITModule` for immediate execution (`marietta run`) and
  `cranelift_object::ObjectModule` to emit native object files (`marietta build`).
- Integer types up to 64 bits map directly to Cranelift's `I8`/`I16`/`I32`/`I64`
  types. Wider integers (up to 2048 bits) are lowered to arrays of `I64` limbs
  with helper functions generated inline (add-with-carry, shift, etc.).
- Float types: `f32` → `F32`, `f64` → `F64`; other float widths are software-
  emulated via helper functions synthesised during code generation.
- The async state-machine structs are lowered to Cranelift stack slots /
  heap-allocated blobs whose layout is computed during IR lowering.
- Function signatures are derived from the type-checked IR; the calling
  convention follows the platform ABI via Cranelift's `isa::CallConv`.

### 10. Runtime (`src/runtime/`)

- A minimal Rust runtime crate linked into every Marietta executable.
- Implements a work-stealing task scheduler for async coroutines and actors.
- Provides `channel::<T>()` returning `(Sender<T>, Receiver<T>)` backed by
  a lock-free MPSC queue.
- Exposes `spawn(actor)`, `send(channel, msg)`, `recv(channel) -> impl Future`.
- Keep the runtime small; no external async runtime dependency in v0.

### 11. Diagnostics & Error Reporting (`src/diagnostics.rs`)

- A `Diagnostic` struct: severity (`Error`/`Warning`/`Note`), `Span`, message string.
- A `DiagnosticSink` collects diagnostics from every pass.
- On output, render a rustc-style snippet: file path, line:col, the source line,
  a caret pointing to the span, and the message.
- Support `--error-format=json` for editor integration.

### 12. Driver (`src/main.rs`)

- Parse CLI arguments: `marietta build <file>`, `marietta run <file>`, `marietta check <file>`.
- Run the pipeline: lex → parse → resolve → type-check → async-transform →
  actor-lower → IR → codegen → link.
- `check` stops after type-checking and prints diagnostics.
- `run` builds to a temp directory and immediately executes the binary.

### 13. Test Suite

- `tests/` directory with `.marietta` source files and expected `.stdout` / `.stderr`
  snapshots.
- `tests/fib256.marietta` — 256-bit Fibonacci using `u256`.
- `tests/matmul.marietta` — matrix multiply over `f32`.
- `tests/actors.marietta` — a counter actor demonstrating RPC.
- `tests/channels.marietta` — producer / consumer with a bounded channel.
- Run via `cargo test`; each test compiles and executes the file and diffs output.

