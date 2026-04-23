/// Three-address IR and lowering pass for the Marietta compiler.
///
/// This module defines a simple SSA-flavoured IR with basic blocks and provides
/// a lowering pass that translates the typed AST into IR.
///
/// # Structure
///
/// Every function lowers to a [`FnIr`] containing a list of [`BasicBlock`]s.
/// Each basic block has a sequence of [`Instruction`]s and ends with a
/// [`Terminator`].
///
/// # Variables (alloca approach)
///
/// Variables are represented as stack-allocation slots: each named variable
/// gets an [`InstKind::Alloc`] instruction at function entry, and every read /
/// write goes through [`InstKind::Load`] / [`InstKind::Store`].  This
/// "pre-SSA" form is correct and is lifted to proper SSA inside Cranelift's
/// `FunctionBuilder` (via its `declare_var` / `use_var` / `def_var` API)
/// during code generation.
///
/// # Wide integers
///
/// Marietta supports integers up to 2048 bits wide.  Any type wider than 64 bits
/// is represented in the IR as [`IrType::WideInt`] and annotated with its limb
/// count (`(bits + 63) / 64` `u64` limbs).  Arithmetic on wide values uses the
/// same [`InstKind::BinOp`] instruction; code generation (step 9) expands each
/// wide operation into a sequence of Cranelift `I64` add-with-carry / shift
/// instructions.

use std::collections::HashMap;

use crate::ast::*;
use crate::resolve::NodeId;
use crate::types::{InferResult, Type};

// ---------------------------------------------------------------------------
// Value and block identifiers
// ---------------------------------------------------------------------------

/// A handle for an SSA value.  Printed as `%N` in IR dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// A handle for a basic block.  Printed as `bb{N}` in IR dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

// ---------------------------------------------------------------------------
// IR type system
// ---------------------------------------------------------------------------

/// A machine-level type used in the IR.
///
/// Derived from [`crate::types::Type`] via [`ir_type_of`].
#[derive(Debug, Clone, PartialEq)]
pub enum IrType {
    // ---- Narrow signed integers (≤ 64 bits) ----
    I8, I16, I32, I64,
    // ---- Narrow unsigned integers (≤ 64 bits) ----
    U8, U16, U32, U64,
    /// An integer wider than 64 bits — stored as `limbs` consecutive `u64`
    /// values (little-endian limb order).
    WideInt { bits: u16, signed: bool },
    F32,
    F64,
    Bool,
    /// Opaque pointer (target pointer size) — used for aggregates, slices,
    /// strings, function references, and channels.
    Ptr,
    /// A `dyn TraitName` fat pointer: (data_ptr, vtable_ptr) stored as two
    /// consecutive pointer-sized words on the caller's stack.  In Cranelift
    /// terms this is passed / stored as a single `I64` that holds the address
    /// of the two-word block.
    Dyn { trait_name: String },
    /// No value produced (void return, `None` literal).
    Void,
}

impl IrType {
    /// Number of 64-bit limbs required to represent this type.
    pub fn limbs(&self) -> usize {
        match self {
            IrType::WideInt { bits, .. } => (*bits as usize + 63) / 64,
            _ => 1,
        }
    }

    /// `true` when this type requires multi-limb arithmetic.
    pub fn is_wide(&self) -> bool {
        matches!(self, IrType::WideInt { .. })
    }
}

/// Map a resolved [`Type`] to an [`IrType`].
pub fn ir_type_of(ty: &Type) -> IrType {
    match ty {
        Type::Int { bits, signed } => {
            if      *bits <=  8 { if *signed { IrType::I8  } else { IrType::U8  } }
            else if *bits <= 16 { if *signed { IrType::I16 } else { IrType::U16 } }
            else if *bits <= 32 { if *signed { IrType::I32 } else { IrType::U32 } }
            else if *bits <= 64 { if *signed { IrType::I64 } else { IrType::U64 } }
            else                { IrType::WideInt { bits: *bits, signed: *signed } }
        }
        Type::Float { bits, .. } => if *bits <= 32 { IrType::F32 } else { IrType::F64 },
        Type::Bool   => IrType::Bool,
        Type::None_  => IrType::Void,
        Type::IntLit   => IrType::I64,   // default: i64
        Type::FloatLit => IrType::F64,   // default: f64
        Type::Dyn(name) => IrType::Dyn { trait_name: name.clone() },
        // Aggregate / pointer-like types — pointer-sized in the IR.
        _ => IrType::Ptr,
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum IrBinOp {
    Add, Sub, Mul, Div, Rem,
    And, Or, Xor, Shl, Shr,
    CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IrUnaryOp {
    Neg,    // arithmetic negation  `-x`
    Not,    // boolean not          `not x`
    BitNot, // bitwise complement   `~x`
}

// ---------------------------------------------------------------------------
// Constant values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ConstVal {
    /// Fits in a signed 128-bit value (covers all integers ≤ 64 bits).
    Int(i128),
    /// Wide integer (> 64 bits): limbs in little-endian `u64` order.
    WideInt(Vec<u64>),
    Float(f64),
    Bool(bool),
    None,
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// A single three-address instruction, annotated with the source slice it
/// originated from for debug info.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction<'src> {
    /// Source slice of the expression / statement that produced this instruction.
    pub src: &'src str,
    pub kind: InstKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InstKind {
    /// `dst = lhs op rhs` — `ty` is the operand type (used to select signed vs
    /// unsigned and integer vs float instructions during code generation).
    BinOp { op: IrBinOp, dst: ValueId, lhs: ValueId, rhs: ValueId, ty: IrType },
    /// `dst = op operand`
    UnaryOp { op: IrUnaryOp, dst: ValueId, operand: ValueId },
    /// `[dst =] func(args…)` — `dst` is `None` for void calls.
    /// Used for indirect / unresolved calls; prefer [`InstKind::DirectCall`].
    Call { dst: Option<ValueId>, func: ValueId, args: Vec<ValueId> },
    /// Direct call to a named function in the same module.
    DirectCall { dst: Option<ValueId>, func_name: String, args: Vec<ValueId> },
    /// Dynamic dispatch through a vtable.
    ///
    /// `fat_ptr` is an SSA value holding the address of a two-word block
    /// `[data_ptr: Ptr, vtable_ptr: Ptr]` on the caller's stack.
    /// `trait_name` identifies the trait so codegen can look up the method
    /// signature from the registered vtables.
    /// `method_idx` is the 0-based index of the method in the trait's vtable.
    /// The data pointer is automatically prepended to `args` before the call.
    VtableCall { dst: Option<ValueId>, fat_ptr: ValueId, trait_name: String, method_idx: usize, args: Vec<ValueId> },
    /// `dst = alloca(ty)` — allocate a stack slot for a mutable variable.
    Alloc { dst: ValueId, ty: IrType },
    /// `dst = *ptr`
    Load { dst: ValueId, ptr: ValueId },
    /// `*ptr = val`
    Store { ptr: ValueId, val: ValueId },
    /// `dst = <constant>`
    Const { dst: ValueId, val: ConstVal, ty: IrType },
    /// `dst = obj.fields[field_idx]`
    GetField { dst: ValueId, obj: ValueId, field_idx: u32 },
    /// `obj.fields[field_idx] = val`
    SetField { obj: ValueId, field_idx: u32, val: ValueId },
    /// `dst = src_val`  (parameter binding or explicit copy)
    Copy { dst: ValueId, src_val: ValueId },
    /// SSA φ-node: `dst = φ[(blk0, v0), (blk1, v1), …]`
    Phi { dst: ValueId, incoming: Vec<(BlockId, ValueId)> },
}

// ---------------------------------------------------------------------------
// Block terminators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    /// Unconditional jump to `target`.
    Jump(BlockId),
    /// `if cond goto then_blk else goto else_blk`
    Branch { cond: ValueId, then_blk: BlockId, else_blk: BlockId },
    /// `return [value]`
    Return(Option<ValueId>),
    /// Unreachable — placeholder for blocks not yet terminated.
    Unreachable,
}

// ---------------------------------------------------------------------------
// Basic block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BasicBlock<'src> {
    pub id: BlockId,
    pub insts: Vec<Instruction<'src>>,
    pub terminator: Terminator,
}

// ---------------------------------------------------------------------------
// Function IR
// ---------------------------------------------------------------------------

/// IR for one function (or method).
#[derive(Debug, Clone)]
pub struct FnIr<'src> {
    /// Source text of the function definition (for debug info).
    pub src: &'src str,
    /// Fully-qualified (potentially mangled) name used in the module.
    ///
    /// For free functions and `impl` block methods this is the same as the
    /// source name.  For `impl Trait for Type` methods the name is mangled
    /// to `{Type}__{method}` so that multiple impls of the same method name
    /// do not collide in the module symbol table.
    pub name: String,
    /// Parameter SSA values and their types.  These are the raw incoming
    /// values; each is immediately stored into its alloca at function entry.
    pub params: Vec<(ValueId, IrType)>,
    /// Return type of the function.
    pub ret_ty: IrType,
    /// Basic blocks, in order; `blocks[0]` is the entry block.
    pub blocks: Vec<BasicBlock<'src>>,
}

// ---------------------------------------------------------------------------
// Module IR
// ---------------------------------------------------------------------------

/// One vtable emitted for an `impl Trait for Type` block.
///
/// The vtable name is `{Type}__{Trait}__vtable`.
/// `methods` lists fully-mangled function names in the same order as the
/// corresponding `TraitDef::methods` list.
#[derive(Debug, Clone)]
pub struct IrVtable {
    pub vtable_name: String,
    pub trait_name: String,
    pub type_name: String,
    /// Mangled function names in trait-method declaration order.
    pub methods: Vec<String>,
}

#[derive(Debug)]
pub struct IrModule<'src> {
    pub functions: Vec<FnIr<'src>>,
    /// One vtable per `impl Trait for Type` block.
    pub vtables: Vec<IrVtable>,
    pub diagnostics: Vec<IrDiagnostic<'src>>,
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct IrDiagnostic<'src> {
    pub src: &'src str,
    pub message: &'static str,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Lower the typed AST to IR.
///
/// Walks every function, method, and actor method in `module`, producing one
/// [`FnIr`] per function.  Type information is taken from `infer_result`.
pub fn lower<'src>(
    source:       &'src str,
    module:       &Module<'src>,
    infer_result: &mut InferResult<'src>,
) -> IrModule<'src> {
    // Eagerly resolve all NodeIds to IrTypes so the lowerer can query types
    // without needing `&mut TypeStore` during the traversal.
    let ir_types: HashMap<NodeId, IrType> = infer_result
        .types
        .iter()
        .map(|(nid, tid)| (*nid, ir_type_of(&infer_result.store.resolve(*tid))))
        .collect();

    // For each Fn-typed node, extract the return type so lower_fn can look it up.
    // An unresolved (Error) return type means no annotation and no return statement → void.
    let fn_ret_types: HashMap<NodeId, IrType> = infer_result
        .types
        .iter()
        .filter_map(|(nid, tid)| {
            if let Type::Fn { ret, .. } = infer_result.store.resolve(*tid) {
                let ir_ret = match infer_result.store.resolve(ret) {
                    Type::None_ | Type::Error => IrType::Void,
                    t => ir_type_of(&t),
                };
                Some((*nid, ir_ret))
            } else {
                None
            }
        })
        .collect();

    // Pre-pass: build vtable method index so dyn call lowering can look up
    // method indices without needing a mutable borrow of the module later.
    let mut vtable_method_index: HashMap<(String, String), usize> = HashMap::new();
    for item in &module.items {
        if let ItemKind::ImplFor(i) = &item.kind {
            for (idx, m) in i.methods.iter().enumerate() {
                vtable_method_index
                    .entry((i.trait_name.to_string(), m.name.to_string()))
                    .or_insert(idx);
            }
        }
    }

    let mut lowerer = Lowerer {
        source,
        ir_types: &ir_types,
        fn_ret_types: &fn_ret_types,
        vtable_method_index: &vtable_method_index,
        diagnostics: Vec::new(),
    };

    let mut functions = Vec::new();
    let mut vtables: Vec<IrVtable> = Vec::new();

    for item in &module.items {
        match &item.kind {
            ItemKind::FunctionDef(f) => {
                functions.push(lowerer.lower_fn(f));
            }
            ItemKind::ImplBlock(ib) => {
                for m in &ib.methods {
                    functions.push(lowerer.lower_fn(m));
                }
            }
            ItemKind::ImplFor(i) => {
                // Mangle each method name: `{Type}__{method}` to avoid
                // collisions when multiple types implement the same trait.
                let mut vtable_methods = Vec::new();
                for m in &i.methods {
                    let mangled = format!("{}__{}" , i.type_name, m.name);
                    let mut fn_ir = lowerer.lower_fn(m);
                    fn_ir.name = mangled.clone();
                    functions.push(fn_ir);
                    vtable_methods.push(mangled);
                }
                // Record the vtable so codegen can construct it.
                let vtable_name = format!("{}__{}__vtable", i.type_name, i.trait_name);
                vtables.push(IrVtable {
                    vtable_name,
                    trait_name: i.trait_name.to_string(),
                    type_name:  i.type_name.to_string(),
                    methods:    vtable_methods,
                });
            }
            ItemKind::TraitDef(_) => {
                // Trait defs are signatures only — no IR emitted.
            }
            ItemKind::ActorDef(ad) => {
                for m in &ad.methods {
                    functions.push(lowerer.lower_fn(m));
                }
            }
            _ => {}
        }
    }

    IrModule { functions, vtables, diagnostics: lowerer.diagnostics }
}

// ---------------------------------------------------------------------------
// Function builder
// ---------------------------------------------------------------------------

/// Mutable state for building one function's IR.
struct FnBuilder<'src> {
    next_val: u32,
    blocks: Vec<BasicBlock<'src>>,
    current: usize,
    /// Maps variable names to their alloca `ValueId` (the pointer).
    var_map: HashMap<&'src str, ValueId>,
    /// Maps variable names to their [`IrType`] — needed for augmented assigns
    /// to select the correct arithmetic instruction (e.g. `udiv` vs `sdiv`).
    var_types: HashMap<&'src str, IrType>,
    /// Target block for `break` statements, if any.
    break_target: Option<BlockId>,
    /// Target block for `continue` statements, if any.
    continue_target: Option<BlockId>,
}

impl<'src> FnBuilder<'src> {
    fn new() -> Self {
        let entry = BasicBlock {
            id: BlockId(0),
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
        };
        FnBuilder {
            next_val: 0,
            blocks: vec![entry],
            current: 0,
            var_map: HashMap::new(),
            var_types: HashMap::new(),
            break_target: None,
            continue_target: None,
        }
    }

    /// Allocate a fresh [`ValueId`].
    fn fresh(&mut self) -> ValueId {
        let id = ValueId(self.next_val);
        self.next_val += 1;
        id
    }

    /// Create a new (empty) basic block and return its id.
    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    /// Switch the "current block" cursor.
    fn set_current(&mut self, id: BlockId) {
        self.current = id.0 as usize;
    }

    fn current_id(&self) -> BlockId {
        self.blocks[self.current].id
    }

    fn current_block(&self) -> &BasicBlock<'src> {
        &self.blocks[self.current]
    }

    fn current_block_mut(&mut self) -> &mut BasicBlock<'src> {
        &mut self.blocks[self.current]
    }

    fn emit(&mut self, src: &'src str, kind: InstKind) {
        self.blocks[self.current].insts.push(Instruction { src, kind });
    }

    /// `true` if the current block already has a final terminator (not Unreachable).
    fn is_terminated(&self) -> bool {
        !matches!(self.current_block().terminator, Terminator::Unreachable)
    }
}

// ---------------------------------------------------------------------------
// Lowerer
// ---------------------------------------------------------------------------

struct Lowerer<'src, 'ir> {
    source:       &'src str,
    ir_types:     &'ir HashMap<NodeId, IrType>,
    /// Maps function-declaration NodeId → return IrType (extracted from Fn types).
    fn_ret_types: &'ir HashMap<NodeId, IrType>,
    /// Maps `(trait_name, method_name)` → vtable slot index.
    /// Populated by `lower()` as it processes `ImplFor` items.
    vtable_method_index: &'ir HashMap<(String, String), usize>,
    diagnostics:  Vec<IrDiagnostic<'src>>,
}

impl<'src, 'ir> Lowerer<'src, 'ir> {
    /// Look up the IR type for the AST node whose source text is `src_slice`.
    fn type_of(&self, src_slice: &'src str) -> IrType {
        let base = self.source.as_ptr() as usize;
        let ptr  = src_slice.as_ptr() as usize;
        let nid  = NodeId(ptr.saturating_sub(base));
        self.ir_types.get(&nid).cloned().unwrap_or(IrType::Void)
    }

    // ------------------------------------------------------------------
    // Function lowering
    // ------------------------------------------------------------------

    fn lower_fn(&mut self, func: &FunctionDef<'src>) -> FnIr<'src> {
        let mut b = FnBuilder::new();

        // 1. Emit parameter SSA values and allocas.
        let mut params = Vec::new();
        for p in &func.params {
            // Parameters are stored in the TypeMap under the full param src slice
            // (e.g. "a: u64"), which is the key used by the type checker's bind().
            let ty = match self.type_of(p.src) {
                IrType::Void => IrType::Ptr, // unresolved param — fall back to ptr
                t => t,
            };
            let param_val = b.fresh();
            params.push((param_val, ty.clone()));

            // Alloca for the param so it can be re-assigned.
            let alloca = b.fresh();
            b.emit(p.src, InstKind::Alloc { dst: alloca, ty: ty.clone() });
            b.emit(p.src, InstKind::Store { ptr: alloca, val: param_val });
            b.var_map.insert(p.name, alloca);
            b.var_types.insert(p.name, ty);
        }

        // 2. Pre-allocate slots for all declared locals (conservative; avoids
        //    forward-reference problems in loops / branches).
        let locals = collect_locals(func.body.as_slice());
        for (name, local_src) in &locals {
            if !b.var_map.contains_key(name) {
                let ty = self.type_of(local_src);
                let alloca = b.fresh();
                b.emit(local_src, InstKind::Alloc { dst: alloca, ty: ty.clone() });
                b.var_map.insert(name, alloca);
                b.var_types.insert(name, ty);
            }
        }

        // Return type is stored inside the Fn type under the function's src NodeId.
        let base = self.source.as_ptr() as usize;
        let fn_ptr = func.src.as_ptr() as usize;
        let fn_nid = NodeId(fn_ptr.saturating_sub(base));
        let ret_ty = self.fn_ret_types.get(&fn_nid).cloned().unwrap_or(IrType::Void);

        // 3. Lower the function body.
        self.lower_stmts(&func.body, &mut b);

        // 4. Ensure every unterminated block has a terminator.
        // For void functions a fall-through is a clean return.  For non-void
        // functions an unterminated block is unreachable dead code (all control
        // paths already returned); leave it as Unreachable so codegen emits a
        // trap rather than a type-incorrect empty return.
        if !b.is_terminated() {
            b.current_block_mut().terminator = if ret_ty == IrType::Void {
                Terminator::Return(None)
            } else {
                Terminator::Unreachable
            };
        }

        FnIr {
            src: func.src,
            name: func.name.to_string(),
            params,
            ret_ty,
            blocks: b.blocks,
        }
    }

    // ------------------------------------------------------------------
    // Statement lowering
    // ------------------------------------------------------------------

    fn lower_stmts(&mut self, stmts: &[Stmt<'src>], b: &mut FnBuilder<'src>) {
        for stmt in stmts {
            self.lower_stmt(stmt, b);
            if b.is_terminated() {
                break;
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt<'src>, b: &mut FnBuilder<'src>) {
        match &stmt.kind {
            StmtKind::VarDecl { name, value, .. } => {
                if let Some(val_expr) = value {
                    let val = self.lower_expr(val_expr, b);
                    if let Some(&ptr) = b.var_map.get(name) {
                        b.emit(stmt.src, InstKind::Store { ptr, val });
                    }
                }
            }

            StmtKind::LetDecl { name, value, .. } => {
                let val = self.lower_expr(value, b);
                if let Some(&ptr) = b.var_map.get(name) {
                    b.emit(stmt.src, InstKind::Store { ptr, val });
                }
            }

            StmtKind::Assign { op, target, value } => {
                let val = self.lower_expr(value, b);
                self.lower_assign(op, target, val, stmt.src, b);
            }

            StmtKind::Return(expr) => {
                let ret_val = expr.as_ref().map(|e| self.lower_expr(e, b));
                b.current_block_mut().terminator = Terminator::Return(ret_val);
            }

            StmtKind::If { branches, else_body } => {
                self.lower_if(branches, else_body, stmt.src, b);
            }

            StmtKind::While { condition, body, .. } => {
                self.lower_while(condition, body, stmt.src, b);
            }

            StmtKind::Expr(e) => {
                self.lower_expr(e, b);
            }

            StmtKind::Pass => {}

            StmtKind::Break => {
                if let Some(target) = b.break_target {
                    b.current_block_mut().terminator = Terminator::Jump(target);
                }
            }

            StmtKind::Continue => {
                if let Some(target) = b.continue_target {
                    b.current_block_mut().terminator = Terminator::Jump(target);
                }
            }

            // For, Import, FromImport, Error — deferred to later passes.
            _ => {}
        }
    }

    // ------------------------------------------------------------------
    // Assignment
    // ------------------------------------------------------------------

    fn lower_assign(
        &mut self,
        op:     &str,
        target: &Expr<'src>,
        val:    ValueId,
        src:    &'src str,
        b:      &mut FnBuilder<'src>,
    ) {
        match &target.kind {
            ExprKind::Name(name) => {
                if let Some(&ptr) = b.var_map.get(name) {
                    let store_val = if op == "=" {
                        val
                    } else {
                        // Augmented: load current, apply binop, store result.
                        // Look up the variable's declared type so we can emit
                        // the correct signed/unsigned/float operation.
                        let ty = b.var_types.get(name).cloned().unwrap_or(IrType::I64);
                        let cur = b.fresh();
                        b.emit(src, InstKind::Load { dst: cur, ptr });
                        let dst = b.fresh();
                        b.emit(src, InstKind::BinOp {
                            op: aug_to_binop(op),
                            dst,
                            lhs: cur,
                            rhs: val,
                            ty,
                        });
                        dst
                    };
                    b.emit(src, InstKind::Store { ptr, val: store_val });
                }
            }

            ExprKind::Attr { obj, attr: _ } => {
                // `self.field = val` — lower the object, then set-field.
                // Full field-index resolution requires struct type info;
                // deferred to a later sub-pass; emit a best-effort SetField.
                let obj_val = self.lower_expr(obj, b);
                b.emit(src, InstKind::SetField { obj: obj_val, field_idx: 0, val });
            }

            _ => {
                self.diagnostics.push(IrDiagnostic {
                    src,
                    message: "complex assignment target not yet lowered",
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // Control flow
    // ------------------------------------------------------------------

    fn lower_if(
        &mut self,
        branches:  &[(Expr<'src>, Vec<Stmt<'src>>)],
        else_body: &[Stmt<'src>],
        src:       &'src str,
        b:         &mut FnBuilder<'src>,
    ) {
        let merge_blk = b.new_block();
        self.lower_if_chain(branches, else_body, merge_blk, src, b);
        b.set_current(merge_blk);
    }

    fn lower_if_chain(
        &mut self,
        branches:  &[(Expr<'src>, Vec<Stmt<'src>>)],
        else_body: &[Stmt<'src>],
        merge:     BlockId,
        src:       &'src str,
        b:         &mut FnBuilder<'src>,
    ) {
        if branches.is_empty() {
            self.lower_stmts(else_body, b);
            if !b.is_terminated() {
                b.current_block_mut().terminator = Terminator::Jump(merge);
            }
            return;
        }

        let (cond_expr, then_body) = &branches[0];
        let then_blk = b.new_block();
        let else_blk = b.new_block();

        let cond = self.lower_expr(cond_expr, b);
        b.current_block_mut().terminator = Terminator::Branch {
            cond,
            then_blk,
            else_blk,
        };

        // Lower then-branch.
        b.set_current(then_blk);
        self.lower_stmts(then_body, b);
        if !b.is_terminated() {
            b.current_block_mut().terminator = Terminator::Jump(merge);
        }

        // Lower remaining elif/else branches in the else block.
        b.set_current(else_blk);
        self.lower_if_chain(&branches[1..], else_body, merge, src, b);
    }

    fn lower_while(
        &mut self,
        condition: &Expr<'src>,
        body:      &[Stmt<'src>],
        _src:      &'src str,
        b:         &mut FnBuilder<'src>,
    ) {
        let cond_blk = b.new_block();
        let body_blk = b.new_block();
        let exit_blk = b.new_block();

        // Fall into the condition check.
        if !b.is_terminated() {
            b.current_block_mut().terminator = Terminator::Jump(cond_blk);
        }

        // Condition block.
        b.set_current(cond_blk);
        let cond = self.lower_expr(condition, b);
        b.current_block_mut().terminator = Terminator::Branch {
            cond,
            then_blk: body_blk,
            else_blk: exit_blk,
        };

        // Body block.
        b.set_current(body_blk);
        let old_break    = b.break_target.replace(exit_blk);
        let old_continue = b.continue_target.replace(cond_blk);
        self.lower_stmts(body, b);
        b.break_target    = old_break;
        b.continue_target = old_continue;
        if !b.is_terminated() {
            b.current_block_mut().terminator = Terminator::Jump(cond_blk);
        }

        // Continue after loop.
        b.set_current(exit_blk);
    }

    // ------------------------------------------------------------------
    // Expression lowering — returns the SSA ValueId for the result.
    // ------------------------------------------------------------------

    fn lower_expr(&mut self, expr: &Expr<'src>, b: &mut FnBuilder<'src>) -> ValueId {
        match &expr.kind {
            ExprKind::IntLiteral(s) => {
                let ty  = self.type_of(expr.src);
                let dst = b.fresh();
                let val = parse_int_literal(s, ty.is_wide());
                b.emit(expr.src, InstKind::Const { dst, val, ty });
                dst
            }

            ExprKind::FloatLiteral(s) => {
                let ty  = self.type_of(expr.src);
                let dst = b.fresh();
                let val = parse_float_literal(s);
                b.emit(expr.src, InstKind::Const { dst, val, ty });
                dst
            }

            ExprKind::BoolLiteral(bv) => {
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Const {
                    dst,
                    val: ConstVal::Bool(*bv),
                    ty: IrType::Bool,
                });
                dst
            }

            ExprKind::NoneLiteral => {
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Const {
                    dst,
                    val: ConstVal::None,
                    ty: IrType::Void,
                });
                dst
            }

            ExprKind::Name(name) => {
                if let Some(&ptr) = b.var_map.get(name) {
                    let dst = b.fresh();
                    b.emit(expr.src, InstKind::Load { dst, ptr });
                    dst
                } else {
                    // Unresolved name — emit a diagnostic and return a fresh
                    // (undefined) value to allow the rest of the function to
                    // lower.
                    self.diagnostics.push(IrDiagnostic {
                        src:     expr.src,
                        message: "undefined variable in IR lowering",
                    });
                    b.fresh()
                }
            }

            ExprKind::BinOp { op, left, right } => {
                let lhs = self.lower_expr(left, b);
                let rhs = self.lower_expr(right, b);
                let dst = b.fresh();
                // Operand type drives signed/unsigned and int/float selection.
                let ty = match self.type_of(left.src) {
                    IrType::Void => self.type_of(expr.src),
                    t => t,
                };
                b.emit(expr.src, InstKind::BinOp {
                    op: ast_binop_to_ir(op),
                    dst,
                    lhs,
                    rhs,
                    ty,
                });
                dst
            }

            ExprKind::UnaryOp { op, operand } => {
                let v   = self.lower_expr(operand, b);
                let dst = b.fresh();
                b.emit(expr.src, InstKind::UnaryOp {
                    op: ast_unop_to_ir(op),
                    dst,
                    operand: v,
                });
                dst
            }

            ExprKind::Call { func, args, kwargs: _ } => {
                let arg_vals: Vec<ValueId> = args.iter()
                    .map(|a| self.lower_expr(a, b))
                    .collect();
                let ty  = self.type_of(expr.src);
                let dst = if ty == IrType::Void { None } else { Some(b.fresh()) };

                // Dyn call: `receiver.method(args)` where receiver has a Dyn type.
                // The receiver is a fat pointer; dispatch through its vtable.
                if let ExprKind::Attr { obj, attr } = &func.kind {
                    let obj_ty = self.type_of(obj.src);
                    if let IrType::Dyn { trait_name } = obj_ty {
                        let key = (trait_name.clone(), attr.to_string());
                        if let Some(&method_idx) = self.vtable_method_index.get(&key) {
                            let fat_ptr = self.lower_expr(obj, b);
                            b.emit(expr.src, InstKind::VtableCall {
                                dst,
                                fat_ptr,
                                trait_name: trait_name.clone(),
                                method_idx,
                                args: arg_vals,
                            });
                            return dst.unwrap_or_else(|| b.fresh());
                        } else {
                            self.diagnostics.push(IrDiagnostic {
                                src:     expr.src,
                                message: "method not found in vtable for dyn call",
                            });
                            return b.fresh();
                        }
                    }
                }

                // Emit a direct call when the callee is a plain name; indirect
                // otherwise.  The `fval` for a name load is still computed so
                // that the Load instruction appears in the IR (it is dead but
                // harmless), keeping all Name expressions consistently lowered.
                if let ExprKind::Name(name) = &func.kind {
                    b.emit(expr.src, InstKind::DirectCall {
                        dst,
                        func_name: name.to_string(),
                        args: arg_vals,
                    });
                } else {
                    let fval = self.lower_expr(func, b);
                    b.emit(expr.src, InstKind::Call { dst, func: fval, args: arg_vals });
                }

                // Return a value even for void calls (callers may ignore it).
                dst.unwrap_or_else(|| b.fresh())
            }

            ExprKind::Attr { obj, attr: _ } => {
                // Struct field read.  Full field-index lookup requires struct
                // type info; field_idx = 0 is a placeholder — resolved in a
                // later sub-pass once struct layouts are computed.
                let obj_val = self.lower_expr(obj, b);
                let dst     = b.fresh();
                b.emit(expr.src, InstKind::GetField {
                    dst,
                    obj: obj_val,
                    field_idx: 0,
                });
                dst
            }

            ExprKind::Await(inner) => {
                // After `async_transform` the await is reified into a
                // state-machine step; at IR level we simply lower the future
                // expression.
                self.lower_expr(inner, b)
            }

            ExprKind::IfExpr { condition, value, alt } => {
                let cond_val  = self.lower_expr(condition, b);
                let then_blk  = b.new_block();
                let else_blk  = b.new_block();
                let merge_blk = b.new_block();
                b.current_block_mut().terminator = Terminator::Branch {
                    cond:     cond_val,
                    then_blk,
                    else_blk,
                };

                b.set_current(then_blk);
                let then_val  = self.lower_expr(value, b);
                let then_end  = b.current_id();
                b.current_block_mut().terminator = Terminator::Jump(merge_blk);

                b.set_current(else_blk);
                let else_val  = self.lower_expr(alt, b);
                let else_end  = b.current_id();
                b.current_block_mut().terminator = Terminator::Jump(merge_blk);

                b.set_current(merge_blk);
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Phi {
                    dst,
                    incoming: vec![(then_end, then_val), (else_end, else_val)],
                });
                dst
            }

            ExprKind::Index { obj, index } => {
                let base  = self.lower_expr(obj, b);
                let idx   = self.lower_expr(index, b);
                let ptr   = b.fresh();
                // Lower as: pointer arithmetic then load.
                b.emit(expr.src, InstKind::BinOp {
                    op: IrBinOp::Add,
                    dst: ptr,
                    lhs: base,
                    rhs: idx,
                    ty: IrType::Ptr,
                });
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Load { dst, ptr });
                dst
            }

            // Tuple / list / array literals — lower all sub-expressions for
            // side effects; the aggregate itself is heap-/stack-allocated by
            // code gen.
            ExprKind::Tuple(elems)
            | ExprKind::List(elems)
            | ExprKind::ArrayLit(elems) => {
                for e in elems {
                    self.lower_expr(e, b);
                }
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Alloc { dst, ty: IrType::Ptr });
                dst
            }

            // Multi-dimensional slice literal — lower start/end expressions.
            ExprKind::MultiSliceLit(ranges) => {
                for (start, end) in ranges {
                    self.lower_expr(start, b);
                    self.lower_expr(end, b);
                }
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Alloc { dst, ty: IrType::Ptr });
                dst
            }

            ExprKind::Lambda { .. } | ExprKind::Error(_) => {
                self.diagnostics.push(IrDiagnostic {
                    src:     expr.src,
                    message: "expression kind not yet lowered to IR",
                });
                b.fresh()
            }

            ExprKind::StringLiteral(_) => {
                let dst = b.fresh();
                b.emit(expr.src, InstKind::Const {
                    dst,
                    val: ConstVal::None, // placeholder; codegen emits a string ref
                    ty:  IrType::Ptr,
                });
                dst
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively collect every `(name, src)` pair declared by VarDecl / LetDecl
/// in `stmts` and nested blocks.
fn collect_locals<'src>(stmts: &[Stmt<'src>]) -> Vec<(&'src str, &'src str)> {
    let mut out = Vec::new();
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::VarDecl { name, .. } | StmtKind::LetDecl { name, .. } => {
                out.push((*name, stmt.src));
            }
            StmtKind::If { branches, else_body } => {
                for (_, body) in branches {
                    out.extend(collect_locals(body));
                }
                out.extend(collect_locals(else_body));
            }
            StmtKind::While { body, else_body, .. } => {
                out.extend(collect_locals(body));
                out.extend(collect_locals(else_body));
            }
            StmtKind::For { body, else_body, .. } => {
                out.extend(collect_locals(body));
                out.extend(collect_locals(else_body));
            }
            _ => {}
        }
    }
    out
}

/// Parse an integer literal (possibly with `_` separators and a type suffix)
/// into a [`ConstVal`].  For wide integers the value is split into `u64` limbs.
fn parse_int_literal(s: &str, wide: bool) -> ConstVal {
    // Strip trailing type suffix (e.g. `_u8`, `_i32`).
    let s = strip_literal_suffix(s);
    // Remove digit-group underscores.
    let digits: String = s.chars().filter(|&c| c != '_').collect();
    let n: u128 = digits.parse().unwrap_or(0);

    if wide {
        let lo = n as u64;
        let hi = (n >> 64) as u64;
        ConstVal::WideInt(vec![lo, hi])
    } else {
        ConstVal::Int(n as i128)
    }
}

/// Parse a float literal (possibly with type suffix) into a [`ConstVal`].
fn parse_float_literal(s: &str) -> ConstVal {
    let s = strip_literal_suffix(s);
    let digits: String = s.chars().filter(|&c| c != '_').collect();
    ConstVal::Float(digits.parse().unwrap_or(0.0))
}

/// Remove a type suffix such as `_u8` or `_f32` from a literal string.
fn strip_literal_suffix(s: &str) -> &str {
    if let Some(idx) = s.rfind('_') {
        let suffix = &s[idx + 1..];
        if suffix.chars().next().map_or(false, |c| c.is_alphabetic()) {
            return &s[..idx];
        }
    }
    s
}

fn ast_binop_to_ir(op: &str) -> IrBinOp {
    match op {
        "+"   => IrBinOp::Add,
        "-"   => IrBinOp::Sub,
        "*"   => IrBinOp::Mul,
        "/"   => IrBinOp::Div,
        "%"   => IrBinOp::Rem,
        "&"   => IrBinOp::And,
        "|"   => IrBinOp::Or,
        "^"   => IrBinOp::Xor,
        "<<"  => IrBinOp::Shl,
        ">>"  => IrBinOp::Shr,
        "=="  => IrBinOp::CmpEq,
        "!="  => IrBinOp::CmpNe,
        "<"   => IrBinOp::CmpLt,
        "<="  => IrBinOp::CmpLe,
        ">"   => IrBinOp::CmpGt,
        ">="  => IrBinOp::CmpGe,
        "and" => IrBinOp::And,
        "or"  => IrBinOp::Or,
        _     => IrBinOp::Add, // fallback
    }
}

fn aug_to_binop(op: &str) -> IrBinOp {
    match op {
        "+=" => IrBinOp::Add,
        "-=" => IrBinOp::Sub,
        "*=" => IrBinOp::Mul,
        "/=" => IrBinOp::Div,
        "%=" => IrBinOp::Rem,
        "&=" => IrBinOp::And,
        "|=" => IrBinOp::Or,
        "^=" => IrBinOp::Xor,
        "<<=" => IrBinOp::Shl,
        ">>=" => IrBinOp::Shr,
        _    => IrBinOp::Add,
    }
}

fn ast_unop_to_ir(op: &str) -> IrUnaryOp {
    match op {
        "-"   => IrUnaryOp::Neg,
        "not" => IrUnaryOp::Not,
        "~"   => IrUnaryOp::BitNot,
        _     => IrUnaryOp::Neg,
    }
}

// ---------------------------------------------------------------------------
// Pretty-printer
// ---------------------------------------------------------------------------
//
// Text format (one instruction per line, indented inside blocks):
//
//   fn add(u64, u64) -> u64:
//   bb0:
//     %0: u64
//     %1 = alloca u64
//     store %1 <- %0
//     %2: u64
//     %3 = alloca u64
//     store %3 <- %2
//     %4 = load %1
//     %5 = load %3
//     %6 = add %4, %5
//     ret %6
//
// Block terminators appear as the last line of each block.

use std::fmt;

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::I8   => write!(f, "i8"),
            IrType::I16  => write!(f, "i16"),
            IrType::I32  => write!(f, "i32"),
            IrType::I64  => write!(f, "i64"),
            IrType::U8   => write!(f, "u8"),
            IrType::U16  => write!(f, "u16"),
            IrType::U32  => write!(f, "u32"),
            IrType::U64  => write!(f, "u64"),
            IrType::WideInt { bits, signed } => {
                write!(f, "{}{bits}", if *signed { 'i' } else { 'u' })
            }
            IrType::F32  => write!(f, "f32"),
            IrType::F64  => write!(f, "f64"),
            IrType::Bool => write!(f, "bool"),
            IrType::Ptr  => write!(f, "ptr"),
            IrType::Dyn { trait_name } => write!(f, "dyn {trait_name}"),
            IrType::Void => write!(f, "void"),
        }
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for IrBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IrBinOp::Add   => "add",
            IrBinOp::Sub   => "sub",
            IrBinOp::Mul   => "mul",
            IrBinOp::Div   => "div",
            IrBinOp::Rem   => "rem",
            IrBinOp::And   => "and",
            IrBinOp::Or    => "or",
            IrBinOp::Xor   => "xor",
            IrBinOp::Shl   => "shl",
            IrBinOp::Shr   => "shr",
            IrBinOp::CmpEq => "eq",
            IrBinOp::CmpNe => "ne",
            IrBinOp::CmpLt => "lt",
            IrBinOp::CmpLe => "le",
            IrBinOp::CmpGt => "gt",
            IrBinOp::CmpGe => "ge",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for IrUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IrUnaryOp::Neg    => "neg",
            IrUnaryOp::Not    => "not",
            IrUnaryOp::BitNot => "bitnot",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for ConstVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstVal::Int(n)       => write!(f, "{n}"),
            ConstVal::WideInt(ls)  => {
                // Print as 0x<hex>, most-significant limb first.
                write!(f, "0x")?;
                for limb in ls.iter().rev() {
                    write!(f, "{limb:016x}")?;
                }
                Ok(())
            }
            ConstVal::Float(v)     => write!(f, "{v}"),
            ConstVal::Bool(b)      => write!(f, "{b}"),
            ConstVal::None         => write!(f, "none"),
        }
    }
}

impl fmt::Display for InstKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstKind::BinOp { op, dst, lhs, rhs, ty } =>
                write!(f, "{dst} = {op}.{ty} {lhs}, {rhs}"),
            InstKind::UnaryOp { op, dst, operand } =>
                write!(f, "{dst} = {op} {operand}"),
            InstKind::Call { dst, func, args } => {
                let arg_list = args.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match dst {
                    Some(d) => write!(f, "{d} = call {func}({arg_list})"),
                    None    => write!(f, "call {func}({arg_list})"),
                }
            }
            InstKind::DirectCall { dst, func_name, args } => {
                let arg_list = args.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match dst {
                    Some(d) => write!(f, "{d} = call @{func_name}({arg_list})"),
                    None    => write!(f, "call @{func_name}({arg_list})"),
                }
            }
            InstKind::VtableCall { dst, fat_ptr, trait_name, method_idx, args } => {
                let arg_list = args.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match dst {
                    Some(d) => write!(f, "{d} = vtable_call {fat_ptr}.{trait_name}[{method_idx}]({arg_list})"),
                    None    => write!(f, "vtable_call {fat_ptr}.{trait_name}[{method_idx}]({arg_list})"),
                }
            }
            InstKind::Alloc { dst, ty }            => write!(f, "{dst} = alloca {ty}"),
            InstKind::Load  { dst, ptr }            => write!(f, "{dst} = load {ptr}"),
            InstKind::Store { ptr, val }            => write!(f, "store {ptr} <- {val}"),
            InstKind::Const { dst, val, ty }        => write!(f, "{dst} = const {ty} {val}"),
            InstKind::GetField { dst, obj, field_idx } =>
                write!(f, "{dst} = getfield {obj}.{field_idx}"),
            InstKind::SetField { obj, field_idx, val } =>
                write!(f, "setfield {obj}.{field_idx} <- {val}"),
            InstKind::Copy  { dst, src_val }        => write!(f, "{dst} = copy {src_val}"),
            InstKind::Phi   { dst, incoming }       => {
                let pairs = incoming.iter()
                    .map(|(blk, val)| format!("[{blk}: {val}]"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{dst} = phi {pairs}")
            }
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Jump(blk)                        => write!(f, "jump {blk}"),
            Terminator::Branch { cond, then_blk, else_blk } =>
                write!(f, "br {cond}, {then_blk}, {else_blk}"),
            Terminator::Return(Some(v))                  => write!(f, "ret {v}"),
            Terminator::Return(None)                     => write!(f, "ret"),
            Terminator::Unreachable                      => write!(f, "unreachable"),
        }
    }
}

impl<'src> fmt::Display for BasicBlock<'src> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.id)?;
        for inst in &self.insts {
            writeln!(f, "  {}", inst.kind)?;
        }
        writeln!(f, "  {}", self.terminator)
    }
}

impl<'src> fmt::Display for FnIr<'src> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let param_types = self.params.iter()
            .map(|(_, ty)| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "fn {}({}) -> {}:", self.name, param_types, self.ret_ty)?;
        for block in &self.blocks {
            write!(f, "{block}")?;
        }
        Ok(())
    }
}

impl<'src> fmt::Display for IrModule<'src> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, func) in self.functions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{func}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, resolve, types};

    fn run(src: &str) -> IrModule<'_> {
        let pr  = parser::parse(src);
        let rr  = resolve::resolve(src, &pr.module);
        let mut ir = types::infer(src, &pr.module, &rr.resolutions);
        lower(src, &pr.module, &mut ir)
    }

    // Helper: find a function by name.
    fn find_fn<'a>(m: &'a IrModule<'a>, name: &str) -> &'a FnIr<'a> {
        m.functions.iter().find(|f| f.name == name).expect("function not found")
    }

    // Helper: collect all InstKinds in a function across all blocks.
    fn all_insts<'a>(f: &'a FnIr<'a>) -> Vec<&'a InstKind> {
        f.blocks.iter().flat_map(|b| b.insts.iter().map(|i| &i.kind)).collect()
    }

    #[test]
    fn empty_module_no_functions() {
        let m = run("pass\n");
        assert!(m.functions.is_empty());
    }

    #[test]
    fn empty_function_single_block() {
        let m = run("def noop():\n    pass\n");
        let f = find_fn(&m, "noop");
        // entry block + Return(None) terminator
        assert_eq!(f.blocks.len(), 1);
        assert_eq!(f.blocks[0].terminator, Terminator::Return(None));
    }

    #[test]
    fn return_constant() {
        let m = run("def forty_two() -> u64:\n    return 42\n");
        let f = find_fn(&m, "forty_two");
        // Should have a Const instruction and Return terminator.
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::Const { .. })));
        let terminators: Vec<_> = f.blocks.iter().map(|b| &b.terminator).collect();
        assert!(terminators.iter().any(|t| matches!(t, Terminator::Return(Some(_)))));
    }

    #[test]
    fn binary_arithmetic_emits_binop() {
        let m = run("def add(a: u64, b: u64) -> u64:\n    return a + b\n");
        let f = find_fn(&m, "add");
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::BinOp { op: IrBinOp::Add, .. })));
    }

    #[test]
    fn params_get_alloca_and_store() {
        let m = run("def id(x: u64) -> u64:\n    return x\n");
        let f = find_fn(&m, "id");
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::Alloc { .. })));
        assert!(insts.iter().any(|k| matches!(k, InstKind::Store { .. })));
        assert!(insts.iter().any(|k| matches!(k, InstKind::Load { .. })));
    }

    #[test]
    fn var_decl_emits_store() {
        let m = run("def f():\n    var x: u64 = 10\n");
        let f = find_fn(&m, "f");
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::Store { .. })));
        assert!(insts.iter().any(|k| matches!(k, InstKind::Const { .. })));
    }

    #[test]
    fn if_else_produces_branch_and_merge() {
        let m = run(
            "def sign(x: u64) -> u64:\n    if x > 0:\n        return 1\n    else:\n        return 0\n",
        );
        let f = find_fn(&m, "sign");
        // Expect: entry block → Branch, then-block, else-block, (merge unreachable or not reached)
        let has_branch = f.blocks.iter().any(|b| matches!(b.terminator, Terminator::Branch { .. }));
        assert!(has_branch, "expected a Branch terminator");
        // At least 3 blocks (entry with branch, then, else)
        assert!(f.blocks.len() >= 3);
    }

    #[test]
    fn while_loop_three_blocks() {
        let m = run("def count():\n    var i: u64 = 0\n    while i < 10:\n        i += 1\n");
        let f = find_fn(&m, "count");
        // entry block → Jump(cond) — cond block → Branch — body block → Jump(cond)
        let has_jump_back = f.blocks.iter().any(|b| {
            // body block jumps back to cond block
            if let Terminator::Jump(target) = b.terminator {
                // cond block should be the branch block
                f.blocks.iter().any(|bb| bb.id == target && matches!(bb.terminator, Terminator::Branch { .. }))
            } else {
                false
            }
        });
        assert!(has_jump_back, "while loop should have a back-edge to the condition block");
        assert!(f.blocks.len() >= 3);
    }

    #[test]
    fn augmented_assign_emits_load_binop_store() {
        let m = run("def f():\n    var x: u64 = 0\n    x += 5\n");
        let f = find_fn(&m, "f");
        let insts = all_insts(f);
        // augmented assign = Load + BinOp(Add) + Store
        let add_count = insts.iter().filter(|k| matches!(k, InstKind::BinOp { op: IrBinOp::Add, .. })).count();
        assert!(add_count >= 1);
        let load_count = insts.iter().filter(|k| matches!(k, InstKind::Load { .. })).count();
        assert!(load_count >= 1);
    }

    #[test]
    fn wide_integer_type_u128() {
        assert_eq!(
            ir_type_of(&Type::Int { bits: 128, signed: false }),
            IrType::WideInt { bits: 128, signed: false }
        );
        assert_eq!(
            IrType::WideInt { bits: 128, signed: false }.limbs(),
            2
        );
    }

    #[test]
    fn wide_integer_type_u256() {
        let ty = ir_type_of(&Type::Int { bits: 256, signed: false });
        assert_eq!(ty, IrType::WideInt { bits: 256, signed: false });
        assert_eq!(ty.limbs(), 4);
    }

    #[test]
    fn narrow_integer_types() {
        assert_eq!(ir_type_of(&Type::Int { bits:  8, signed: true  }), IrType::I8);
        assert_eq!(ir_type_of(&Type::Int { bits: 16, signed: false }), IrType::U16);
        assert_eq!(ir_type_of(&Type::Int { bits: 32, signed: true  }), IrType::I32);
        assert_eq!(ir_type_of(&Type::Int { bits: 64, signed: false }), IrType::U64);
    }

    #[test]
    fn float_types() {
        assert_eq!(ir_type_of(&Type::Float { bits: 32, mantissa: 23 }), IrType::F32);
        assert_eq!(ir_type_of(&Type::Float { bits: 64, mantissa: 52 }), IrType::F64);
    }

    #[test]
    fn actor_methods_lowered() {
        let m = run(
            "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n",
        );
        let f = find_fn(&m, "increment");
        // self param should be present
        assert_eq!(f.params.len(), 1);
        // self param type is Ptr
        assert_eq!(f.params[0].1, IrType::Ptr);
    }

    #[test]
    fn impl_methods_lowered() {
        let src = "struct Point:\n    x: u64\n    y: u64\n\nimpl Point:\n    def sum(self) -> u64:\n        return self.x + self.y\n";
        let m = run(src);
        let f = find_fn(&m, "sum");
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::BinOp { op: IrBinOp::Add, .. })));
    }

    #[test]
    fn multiple_functions_in_module() {
        let m = run("def a():\n    pass\n\ndef b():\n    pass\n");
        assert_eq!(m.functions.len(), 2);
    }

    #[test]
    fn call_emits_call_instruction() {
        let m = run("def f():\n    print(42)\n");
        let f = find_fn(&m, "f");
        let insts = all_insts(f);
        // Named calls are lowered to DirectCall; the generic Call variant is
        // only emitted for indirect / first-class function values.
        assert!(
            insts.iter().any(|k| matches!(k, InstKind::DirectCall { .. }
                                            | InstKind::Call { .. })),
            "expected a Call or DirectCall instruction"
        );
    }

    #[test]
    fn unary_negation() {
        let m = run("def neg(x: u64) -> u64:\n    return -x\n");
        let f = find_fn(&m, "neg");
        let insts = all_insts(f);
        assert!(insts.iter().any(|k| matches!(k, InstKind::UnaryOp { op: IrUnaryOp::Neg, .. })));
    }

    #[test]
    fn strip_literal_suffix_works() {
        assert_eq!(strip_literal_suffix("42_u8"),   "42");
        assert_eq!(strip_literal_suffix("1_000_i32"), "1_000");
        assert_eq!(strip_literal_suffix("99"),       "99");
        assert_eq!(strip_literal_suffix("3_14_f32"), "3_14");
    }

    #[test]
    fn no_ir_diagnostics_for_simple_fn() {
        let m = run("def add(a: u64, b: u64) -> u64:\n    return a + b\n");
        assert!(m.diagnostics.is_empty(), "unexpected diagnostics: {:?}", m.diagnostics);
    }

    // ------------------------------------------------------------------
    // Pretty-printer snapshot tests
    // ------------------------------------------------------------------

    #[test]
    fn pp_binary_arithmetic() {
        let m = run("def add(a: u64, b: u64) -> u64:\n    return a + b\n");
        let f = find_fn(&m, "add");
        let expected = "\
fn add(u64, u64) -> u64:
bb0:
  %1 = alloca u64
  store %1 <- %0
  %3 = alloca u64
  store %3 <- %2
  %4 = load %1
  %5 = load %3
  %6 = add.u64 %4, %5
  ret %6
";
        assert_eq!(format!("{f}"), expected);
    }

    #[test]
    fn pp_while_loop() {
        let m = run("def count():\n    var i: u64 = 0\n    while i < 10:\n        i += 1\n");
        let f = find_fn(&m, "count");
        let expected = "\
fn count() -> void:
bb0:
  %0 = alloca u64
  %1 = const u64 0
  store %0 <- %1
  jump bb1
bb1:
  %2 = load %0
  %3 = const u64 10
  %4 = lt.bool %2, %3
  br %4, bb2, bb3
bb2:
  %5 = const u64 1
  %6 = load %0
  %7 = add.u64 %6, %5
  store %0 <- %7
  jump bb1
bb3:
  ret
";
        assert_eq!(format!("{f}"), expected);
    }

    #[test]
    fn pp_if_else() {
        let m = run("def abs(x: i64) -> i64:\n    if x < 0:\n        return -x\n    else:\n        return x\n");
        let f = find_fn(&m, "abs");
        let expected = "\
fn abs(i64) -> i64:
bb0:
  %1 = alloca i64
  store %1 <- %0
  %2 = load %1
  %3 = const i64 0
  %4 = lt.bool %2, %3
  br %4, bb2, bb3
bb1:
  unreachable
bb2:
  %5 = load %1
  %6 = neg %5
  ret %6
bb3:
  %7 = load %1
  ret %7
";
        assert_eq!(format!("{f}"), expected);
    }

    #[test]
    fn pp_actor_increment() {
        let m = run(
            "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n\n    def get(self) -> u64:\n        return self.count\n",
        );
        let f = find_fn(&m, "increment");
        let expected = "\
fn increment(ptr) -> void:
bb0:
  %1 = alloca ptr
  store %1 <- %0
  %2 = const i64 1
  %3 = load %1
  setfield %3.0 <- %2
  ret
";
        assert_eq!(format!("{f}"), expected);
    }

    #[test]
    fn pp_actor_getter() {
        let m = run(
            "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n\n    def get(self) -> u64:\n        return self.count\n",
        );
        let f = find_fn(&m, "get");
        let expected = "\
fn get(ptr) -> u64:
bb0:
  %1 = alloca ptr
  store %1 <- %0
  %2 = load %1
  %3 = getfield %2.0
  ret %3
";
        assert_eq!(format!("{f}"), expected);
    }

    #[test]
    fn pp_ternary_phi() {
        let m = run(
            "def clamp(x: u64, lo: u64, hi: u64) -> u64:\n    return lo if x < lo else hi if x > hi else x\n",
        );
        let f = find_fn(&m, "clamp");
        let expected = "\
fn clamp(u64, u64, u64) -> u64:
bb0:
  %1 = alloca u64
  store %1 <- %0
  %3 = alloca u64
  store %3 <- %2
  %5 = alloca u64
  store %5 <- %4
  %6 = load %1
  %7 = load %3
  %8 = lt.bool %6, %7
  br %8, bb1, bb2
bb1:
  %9 = load %3
  jump bb3
bb2:
  %10 = load %1
  %11 = load %5
  %12 = gt.bool %10, %11
  br %12, bb4, bb5
bb3:
  %16 = phi [bb1: %9], [bb6: %15]
  ret %16
bb4:
  %13 = load %5
  jump bb6
bb5:
  %14 = load %1
  jump bb6
bb6:
  %15 = phi [bb4: %13], [bb5: %14]
  jump bb3
";
        assert_eq!(format!("{f}"), expected);
    }
}
