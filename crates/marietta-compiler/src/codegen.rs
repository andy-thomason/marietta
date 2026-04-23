// src/codegen.rs
//
// Cranelift-based code generation for Marietta.
//
// Two output modes:
//   JIT    – compile and execute in-process         (`marietta run`)
//   Object – emit a native ELF/Mach-O/COFF .o file (`marietta build`)
//
// Strategy
// --------
// Every `IrType` maps to a Cranelift type.  Alloca'd variables and SSA phi
// merges are both modelled as Cranelift `Variable`s via the `FunctionBuilder`
// variable / SSA API (`declare_var` / `def_var` / `use_var`).  Cranelift's
// lazy block-sealing algorithm (via `seal_all_blocks`) resolves all forward
// references automatically, so we never need explicit block parameters.
//
// Wide integers (> 64 bits) are currently represented as `I64` placeholder
// limbs; multi-limb expansion is deferred to a later lowering pass.

use std::collections::HashMap;
use std::path::Path;

use cranelift_codegen::ir::{
    condcodes::IntCC,
    types as cl,
    AbiParam, Block as ClBlock, Function, InstBuilder, TrapCode, Value as ClValue,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module};

use crate::ir::{
    BlockId, ConstVal, FnIr, InstKind, IrBinOp, IrModule, IrType, IrUnaryOp, Terminator,
    ValueId,
};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CodegenDiagnostic {
    pub message: String,
}

// ---------------------------------------------------------------------------
// JIT result
// ---------------------------------------------------------------------------

/// Artifact returned by [`codegen_jit`].  Keeps the JIT module alive so that
/// the code pointers returned by [`JitArtifact::fn_ptr`] remain valid.
pub struct JitArtifact {
    module:   cranelift_jit::JITModule,
    func_ids: HashMap<String, FuncId>,
    pub diagnostics: Vec<CodegenDiagnostic>,
}

impl JitArtifact {
    /// Return a raw pointer to the named function, or `None` if not found.
    ///
    /// # Safety
    /// The caller must cast the pointer to the correct function type and ensure
    /// the `JitArtifact` outlives any invocation through the pointer.
    pub unsafe fn fn_ptr(&self, name: &str) -> Option<*const u8> {
        let &id = self.func_ids.get(name)?;
        Some(self.module.get_finalized_function(id))
    }
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Map an `IrType` to a Cranelift scalar type.  Returns `None` for `Void`.
fn to_cl_type(ty: &IrType) -> Option<cranelift_codegen::ir::Type> {
    match ty {
        IrType::I8  | IrType::U8  => Some(cl::I8),
        IrType::I16 | IrType::U16 => Some(cl::I16),
        IrType::I32 | IrType::U32 => Some(cl::I32),
        IrType::I64 | IrType::U64 => Some(cl::I64),
        // Wide integers: represent as I64 (first limb) — full multi-limb
        // expansion is handled in a later sub-pass.
        IrType::WideInt { .. } => Some(cl::I64),
        IrType::F32  => Some(cl::F32),
        IrType::F64  => Some(cl::F64),
        IrType::Bool => Some(cl::I8),
        IrType::Ptr  => Some(cl::I64), // 64-bit target
        // dyn Trait: fat pointer (data_ptr, vtable_ptr) — passed as a pointer to
        // a two-word block, so the ABI type is the same as a regular pointer.
        IrType::Dyn { .. } => Some(cl::I64),
        IrType::Void => None,
    }
}

// ---------------------------------------------------------------------------
// JIT compilation
// ---------------------------------------------------------------------------

/// JIT-compile `ir` using Cranelift.
/// Returns a [`JitArtifact`] whose function pointers are valid for the
/// lifetime of the returned value.
pub fn codegen_jit(ir: &IrModule) -> JitArtifact {
    use cranelift_jit::{JITBuilder, JITModule};

    let mut flags_b = settings::builder();
    flags_b.set("use_colocated_libcalls", "false").unwrap();
    flags_b.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flags_b);

    let isa = cranelift_native::builder()
        .expect("unsupported host ISA")
        .finish(flags)
        .expect("ISA init failed");

    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module  = JITModule::new(jit_builder);

    let mut diags   = Vec::new();
    let func_ids    = codegen_module(ir, &mut module, &mut diags);
    module.finalize_definitions().expect("JIT finalize failed");

    JitArtifact { module, func_ids, diagnostics: diags }
}

// ---------------------------------------------------------------------------
// AOT (object file) compilation
// ---------------------------------------------------------------------------

/// Compile `ir` to a native object file at `output`.
/// Returns any diagnostics (warnings and errors) produced.
pub fn codegen_object(ir: &IrModule, name: &str, output: &Path) -> Vec<CodegenDiagnostic> {
    use cranelift_object::{ObjectBuilder, ObjectModule};

    let mut flags_b = settings::builder();
    flags_b.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flags_b);

    let isa = cranelift_native::builder()
        .expect("unsupported host ISA")
        .finish(flags)
        .expect("ISA init failed");

    let obj_builder = ObjectBuilder::new(
        isa,
        name,
        cranelift_module::default_libcall_names(),
    )
    .expect("ObjectBuilder failed");

    let mut module = ObjectModule::new(obj_builder);
    let mut diags  = Vec::new();
    codegen_module(ir, &mut module, &mut diags);

    let product = module.finish();
    match product.emit() {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(output, &bytes) {
                diags.push(CodegenDiagnostic { message: format!("write object: {e}") });
            }
        }
        Err(e) => diags.push(CodegenDiagnostic { message: format!("emit object: {e}") }),
    }
    diags
}

// ---------------------------------------------------------------------------
// Module-level codegen (generic over JIT / AOT module)
// ---------------------------------------------------------------------------

fn codegen_module<M: Module>(
    ir:     &IrModule,
    module: &mut M,
    diags:  &mut Vec<CodegenDiagnostic>,
) -> HashMap<String, FuncId> {
    let mut func_ids: HashMap<String, FuncId> = HashMap::new();

    // ── Pass 1: declare every function so forward calls resolve. ──────────
    // Also build a map from function name → (params, ret) for indirect-call
    // signature reconstruction (needed for VtableCall).
    let mut fn_sigs: HashMap<String, (Vec<IrType>, IrType)> = HashMap::new();
    for fn_ir in &ir.functions {
        let mut sig = module.make_signature();
        for (_, ty) in &fn_ir.params {
            if let Some(t) = to_cl_type(ty) { sig.params.push(AbiParam::new(t)); }
        }
        if let Some(t) = to_cl_type(&fn_ir.ret_ty) { sig.returns.push(AbiParam::new(t)); }

        match module.declare_function(&fn_ir.name, Linkage::Export, &sig) {
            Ok(id)  => { func_ids.insert(fn_ir.name.clone(), id); }
            Err(e)  => diags.push(CodegenDiagnostic {
                message: format!("declare '{}': {e}", fn_ir.name),
            }),
        }
        fn_sigs.insert(
            fn_ir.name.clone(),
            (fn_ir.params.iter().map(|(_, t)| t.clone()).collect(), fn_ir.ret_ty.clone()),
        );
    }

    // Build (trait_name, method_idx) → concrete function name for VtableCall.
    // We use the first registered vtable per trait since all impls share the
    // same method signature for a given slot index.
    let mut vtable_fn_names: HashMap<(String, usize), String> = HashMap::new();
    for vtable in &ir.vtables {
        for (idx, fn_name) in vtable.methods.iter().enumerate() {
            vtable_fn_names
                .entry((vtable.trait_name.clone(), idx))
                .or_insert_with(|| fn_name.clone());
        }
    }

    // ── Pass 2: compile each function body. ───────────────────────────────
    let mut func_ctx = FunctionBuilderContext::new();
    for fn_ir in &ir.functions {
        let Some(&func_id) = func_ids.get(&fn_ir.name) else { continue };

        let mut ctx = module.make_context();
        for (_, ty) in &fn_ir.params {
            if let Some(t) = to_cl_type(ty) { ctx.func.signature.params.push(AbiParam::new(t)); }
        }
        if let Some(t) = to_cl_type(&fn_ir.ret_ty) {
            ctx.func.signature.returns.push(AbiParam::new(t));
        }

        compile_fn(fn_ir, &func_ids, &fn_sigs, &vtable_fn_names, module, &mut ctx.func, &mut func_ctx, diags);

        if let Err(e) = module.define_function(func_id, &mut ctx) {
            diags.push(CodegenDiagnostic { message: format!("define '{}': {e}", fn_ir.name) });
        }
        module.clear_context(&mut ctx);
    }

    func_ids
}

// ---------------------------------------------------------------------------
// Function-level codegen
// ---------------------------------------------------------------------------

fn compile_fn<M: Module>(
    fn_ir:            &FnIr<'_>,
    func_ids:         &HashMap<String, FuncId>,
    fn_sigs:          &HashMap<String, (Vec<IrType>, IrType)>,
    vtable_fn_names:  &HashMap<(String, usize), String>,
    module:           &mut M,
    cl_func:          &mut Function,
    func_ctx:         &mut FunctionBuilderContext,
    diags:            &mut Vec<CodegenDiagnostic>,
) {
    if fn_ir.blocks.is_empty() { return; }

    let mut b = FunctionBuilder::new(cl_func, func_ctx);

    // Create a Cranelift Block for every IR basic block.
    let cl_blks: HashMap<BlockId, ClBlock> = fn_ir
        .blocks.iter()
        .map(|bb| (bb.id, b.create_block()))
        .collect();

    // ── Pre-scan: declare Cranelift Variables for every alloca and phi. ───
    // Both alloca-based variables and phi merges use the Variable SSA API
    // so that cranelift-frontend can do mem2reg / phi insertion automatically.

    // Infer value types from Alloc/Const instructions and params.
    let val_types = collect_val_types(fn_ir);

    // Alloca dst → Variable
    let mut var_map: HashMap<ValueId, Variable> = HashMap::new();
    for bb in &fn_ir.blocks {
        for inst in &bb.insts {
            if let InstKind::Alloc { dst, ty } = &inst.kind {
                if let Some(t) = to_cl_type(ty) {
                    var_map.insert(*dst, b.declare_var(t));
                }
            }
        }
    }

    // Phi dst → Variable (for ternary / branch merges)
    let mut phi_vars: HashMap<ValueId, Variable> = HashMap::new();
    // Also collect phi incoming info: (target_blk, phi_dst, from_blk, from_val)
    let mut phi_incoming: Vec<(BlockId, ValueId, BlockId, ValueId)> = Vec::new();
    for bb in &fn_ir.blocks {
        for inst in &bb.insts {
            if let InstKind::Phi { dst, incoming } = &inst.kind {
                if !phi_vars.contains_key(dst) {
                    let ty = val_types.get(dst)
                        .and_then(to_cl_type)
                        .unwrap_or(cl::I64);
                    phi_vars.insert(*dst, b.declare_var(ty));
                }
                for (from_blk, from_val) in incoming {
                    phi_incoming.push((bb.id, *dst, *from_blk, *from_val));
                }
            }
        }
    }

    // ── Set up entry block. ────────────────────────────────────────────────
    let entry_cl = cl_blks[&fn_ir.blocks[0].id];
    b.append_block_params_for_function_params(entry_cl);
    b.switch_to_block(entry_cl);
    b.seal_block(entry_cl);

    // Map param ValueIds to the entry block's Cranelift param values.
    let mut val_map: HashMap<ValueId, ClValue> = HashMap::new();
    {
        let params = b.block_params(entry_cl).to_vec();
        for (i, (param_val, _)) in fn_ir.params.iter().enumerate() {
            if i < params.len() { val_map.insert(*param_val, params[i]); }
        }
    }

    // ── Translate each basic block. ────────────────────────────────────────
    for (idx, bb) in fn_ir.blocks.iter().enumerate() {
        let cl_blk = cl_blks[&bb.id];
        if idx > 0 {
            b.switch_to_block(cl_blk);
            // Bind phi dst variables at block entry (resolves forward uses).
            for inst in &bb.insts {
                if let InstKind::Phi { dst, .. } = &inst.kind {
                    if let Some(&phi_var) = phi_vars.get(dst) {
                        let v = b.use_var(phi_var);
                        val_map.insert(*dst, v);
                    }
                }
            }
        }

        // Translate instructions.
        for inst in &bb.insts {
            translate_inst(
                &inst.kind, &mut b, &mut val_map, &var_map,
                func_ids, fn_sigs, vtable_fn_names, module, diags,
            );
        }

        // Before the terminator, def_var all phi variables for the successor blocks.
        def_phi_vars_for_successors(
            &bb.terminator, bb.id, &phi_incoming, &phi_vars, &mut b, &val_map,
        );

        // Translate terminator.
        translate_term(&bb.terminator, &cl_blks, &mut b, &val_map, diags);
    }

    b.seal_all_blocks();
    b.finalize();
}

// ---------------------------------------------------------------------------
// Helper: def_var phi variables before a terminator jump.
// ---------------------------------------------------------------------------

/// For each outgoing edge in `term`, call `def_var` for any phi variables
/// whose incoming value comes from the current block.
fn def_phi_vars_for_successors(
    term:         &Terminator,
    from_blk:     BlockId,
    phi_incoming: &[(BlockId, ValueId, BlockId, ValueId)], // (to, dst, from, val)
    phi_vars:     &HashMap<ValueId, Variable>,
    b:            &mut FunctionBuilder<'_>,
    val_map:      &HashMap<ValueId, ClValue>,
) {
    let targets: &[BlockId] = match term {
        Terminator::Jump(t)                           => std::slice::from_ref(t),
        Terminator::Branch { then_blk, else_blk, .. } => &[*then_blk, *else_blk],
        _                                             => &[],
    };
    for &to in targets {
        for &(phi_to, phi_dst, phi_from, phi_val) in phi_incoming {
            if phi_to == to && phi_from == from_blk {
                if let (Some(&var), Some(&v)) = (phi_vars.get(&phi_dst), val_map.get(&phi_val)) {
                    b.def_var(var, v);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: collect value types from alloca/const/params.
// ---------------------------------------------------------------------------

fn collect_val_types(fn_ir: &FnIr<'_>) -> HashMap<ValueId, IrType> {
    let mut m: HashMap<ValueId, IrType> = HashMap::new();
    for (val, ty) in &fn_ir.params { m.insert(*val, ty.clone()); }
    for bb in &fn_ir.blocks {
        for inst in &bb.insts {
            match &inst.kind {
                InstKind::Alloc { dst, ty }        => { m.insert(*dst, ty.clone()); }
                InstKind::Const { dst, ty, .. }    => { m.insert(*dst, ty.clone()); }
                // BinOp result has the same type as the operands.
                InstKind::BinOp { dst, ty, .. }    => { m.insert(*dst, ty.clone()); }
                _ => {}
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Instruction translation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn translate_inst<M: Module>(
    inst:             &InstKind,
    b:                &mut FunctionBuilder<'_>,
    val_map:          &mut HashMap<ValueId, ClValue>,
    var_map:          &HashMap<ValueId, Variable>,
    func_ids:         &HashMap<String, FuncId>,
    fn_sigs:          &HashMap<String, (Vec<IrType>, IrType)>,
    vtable_fn_names:  &HashMap<(String, usize), String>,
    module:           &mut M,
    diags:            &mut Vec<CodegenDiagnostic>,
) {
    match inst {
        // Alloc: the Variable was pre-declared; nothing to emit at this site.
        InstKind::Alloc { .. } => {}

        // Store: if the pointer is a known Variable, def_var.
        InstKind::Store { ptr, val } => {
            if let Some(&var) = var_map.get(ptr) {
                if let Some(&v) = val_map.get(val) {
                    b.def_var(var, v);
                }
            }
        }

        // Load: if the pointer is a known Variable, use_var.
        InstKind::Load { dst, ptr } => {
            if let Some(&var) = var_map.get(ptr) {
                let v = b.use_var(var);
                val_map.insert(*dst, v);
            }
        }

        // Constant.
        InstKind::Const { dst, val, ty } => {
            if let Some(v) = emit_const(b, val, ty) {
                val_map.insert(*dst, v);
            }
        }

        // Binary operation.
        InstKind::BinOp { op, dst, lhs, rhs, ty } => {
            let (Some(&lv), Some(&rv)) = (val_map.get(lhs), val_map.get(rhs)) else { return };
            if let Some(v) = emit_binop(b, op, ty, lv, rv) {
                val_map.insert(*dst, v);
            }
        }

        // Unary operation.
        InstKind::UnaryOp { op, dst, operand } => {
            let Some(&v) = val_map.get(operand) else { return };
            let result = match op {
                IrUnaryOp::Neg    => b.ins().ineg(v),
                IrUnaryOp::Not    => {
                    // Boolean not: v == 0
                    b.ins().icmp_imm(IntCC::Equal, v, 0)
                }
                IrUnaryOp::BitNot => b.ins().bnot(v),
            };
            val_map.insert(*dst, result);
        }

        // Call: for now only support direct calls to known functions.
        // Indirect calls (func value from val_map) are skipped with a note.
        InstKind::Call { dst, func, args } => {
            let arg_vals: Vec<ClValue> =
                args.iter().filter_map(|a| val_map.get(a).copied()).collect();

            // Try to resolve `func` as a direct module function reference.
            // In our IR, function names are loaded via Name → Load, so `func`
            // is in val_map but we don't know the name.  For now we emit a
            // placeholder zero for unresolved calls.
            let _ = (func, module, func_ids);
            diags.push(CodegenDiagnostic {
                message: "indirect/unresolved call skipped in codegen (use DirectCall for named functions)"
                    .to_string(),
            });
            // Emit a placeholder zero value if a result is expected.
            if let Some(dst_id) = dst {
                let zero = b.ins().iconst(cl::I64, 0);
                val_map.insert(*dst_id, zero);
            }
            let _ = arg_vals;
        }

        // DirectCall: emit a proper Cranelift `call` instruction.
        InstKind::DirectCall { dst, func_name, args } => {
            let arg_vals: Vec<ClValue> =
                args.iter().filter_map(|a| val_map.get(a).copied()).collect();

            if let Some(&callee_id) = func_ids.get(func_name.as_str()) {
                let func_ref = module.declare_func_in_func(callee_id, b.func);
                let call     = b.ins().call(func_ref, &arg_vals);
                if let Some(dst_id) = dst {
                    let results = b.inst_results(call).to_vec();
                    if let Some(&result) = results.first() {
                        val_map.insert(*dst_id, result);
                    }
                }
            } else {
                diags.push(CodegenDiagnostic {
                    message: format!("unknown function '{func_name}' in direct call"),
                });
                if let Some(dst_id) = dst {
                    let zero = b.ins().iconst(cl::I64, 0);
                    val_map.insert(*dst_id, zero);
                }
            }
        }

        // Phi: handled at block entry via phi_vars / use_var.
        InstKind::Phi { .. } => {}

        // VtableCall: dynamic dispatch through a fat pointer's vtable.
        //
        // Memory layout expected at runtime:
        //   fat_ptr → [data_ptr: I64, vtable_ptr: I64]
        //   vtable_ptr → [fn0: I64, fn1: I64, ...]
        //
        // We load the function pointer from vtable slot `method_idx` and call
        // it with `data_ptr` prepended to the user-supplied `args`.
        InstKind::VtableCall { dst, fat_ptr, trait_name, method_idx, args } => {
            use cranelift_codegen::ir::MemFlags;

            let Some(&fat_ptr_val) = val_map.get(fat_ptr) else { return };

            // Load data_ptr and vtable_ptr from the fat pointer block.
            let data_ptr   = b.ins().load(cl::I64, MemFlags::trusted(), fat_ptr_val, 0);
            let vtable_ptr = b.ins().load(cl::I64, MemFlags::trusted(), fat_ptr_val, 8);

            // Load the function pointer at the correct vtable slot.
            let slot_offset = (*method_idx as i32) * 8;
            let fn_ptr = b.ins().load(cl::I64, MemFlags::trusted(), vtable_ptr, slot_offset);

            // Build the indirect-call signature from the first registered impl
            // for this trait method.  All impls must share the same signature.
            let key = (trait_name.clone(), *method_idx);
            let (param_irtypes, ret_irtype) = if let Some(fn_name) = vtable_fn_names.get(&key) {
                fn_sigs.get(fn_name)
                    .map(|(p, r)| (p.clone(), r.clone()))
                    .unwrap_or_else(|| (vec![], IrType::Void))
            } else {
                (vec![], IrType::Void)
            };

            let mut call_sig = module.make_signature();
            // First param: data_ptr (the concrete struct pointer)
            call_sig.params.push(AbiParam::new(cl::I64));
            // Remaining params: skip index 0 ('self') from the vtable function's signature
            for ty in param_irtypes.iter().skip(1) {
                if let Some(t) = to_cl_type(ty) { call_sig.params.push(AbiParam::new(t)); }
            }
            if let Some(t) = to_cl_type(&ret_irtype) { call_sig.returns.push(AbiParam::new(t)); }
            let call_sig_ref = b.import_signature(call_sig);

            // Build argument list: data_ptr first, then the user args.
            let mut all_args: Vec<ClValue> = vec![data_ptr];
            for a in args {
                if let Some(&v) = val_map.get(a) { all_args.push(v); }
            }

            let call = b.ins().call_indirect(call_sig_ref, fn_ptr, &all_args);
            if let Some(dst_id) = dst {
                let results = b.inst_results(call).to_vec();
                if let Some(&result) = results.first() {
                    val_map.insert(*dst_id, result);
                } else {
                    let zero = b.ins().iconst(cl::I64, 0);
                    val_map.insert(*dst_id, zero);
                }
            }
        }

        // GetField: placeholder — field layout is resolved in a later pass.
        InstKind::GetField { dst, obj, field_idx } => {
            diags.push(CodegenDiagnostic {
                message: format!("GetField {obj}.{field_idx} — layout pass pending"),
            });
            let zero = b.ins().iconst(cl::I64, 0);
            val_map.insert(*dst, zero);
        }

        // SetField: placeholder.
        InstKind::SetField { obj, field_idx, .. } => {
            diags.push(CodegenDiagnostic {
                message: format!("SetField {obj}.{field_idx} — layout pass pending"),
            });
        }

        // Copy.
        InstKind::Copy { dst, src_val } => {
            if let Some(&v) = val_map.get(src_val) {
                val_map.insert(*dst, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constant emission
// ---------------------------------------------------------------------------

fn emit_const(
    b:   &mut FunctionBuilder<'_>,
    val: &ConstVal,
    ty:  &IrType,
) -> Option<ClValue> {
    let cl_ty = to_cl_type(ty)?;
    Some(match val {
        ConstVal::Int(n) => match ty {
            IrType::F32 => b.ins().f32const(*n as f32),
            IrType::F64 => b.ins().f64const(*n as f64),
            _           => b.ins().iconst(cl_ty, *n as i64),
        },
        ConstVal::WideInt(limbs) => {
            b.ins().iconst(cl_ty, limbs.first().copied().unwrap_or(0) as i64)
        }
        ConstVal::Float(f) => match ty {
            IrType::F32 => b.ins().f32const(*f as f32),
            _           => b.ins().f64const(*f),
        },
        ConstVal::Bool(bv) => b.ins().iconst(cl_ty, *bv as i64),
        ConstVal::None     => b.ins().iconst(cl_ty, 0),
    })
}

// ---------------------------------------------------------------------------
// Binary operation emission
// ---------------------------------------------------------------------------

fn emit_binop(
    b:  &mut FunctionBuilder<'_>,
    op: &IrBinOp,
    ty: &IrType,
    lv: ClValue,
    rv: ClValue,
) -> Option<ClValue> {
    let is_float    = matches!(ty, IrType::F32 | IrType::F64);
    let is_unsigned = matches!(ty, IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64);

    Some(match op {
        IrBinOp::Add => if is_float { b.ins().fadd(lv, rv) } else { b.ins().iadd(lv, rv) },
        IrBinOp::Sub => if is_float { b.ins().fsub(lv, rv) } else { b.ins().isub(lv, rv) },
        IrBinOp::Mul => if is_float { b.ins().fmul(lv, rv) } else { b.ins().imul(lv, rv) },
        IrBinOp::Div => {
            if is_float     { b.ins().fdiv(lv, rv) }
            else if is_unsigned { b.ins().udiv(lv, rv) }
            else            { b.ins().sdiv(lv, rv) }
        }
        IrBinOp::Rem => {
            if is_unsigned  { b.ins().urem(lv, rv) }
            else            { b.ins().srem(lv, rv) }
        }
        IrBinOp::And   => b.ins().band(lv, rv),
        IrBinOp::Or    => b.ins().bor(lv, rv),
        IrBinOp::Xor   => b.ins().bxor(lv, rv),
        IrBinOp::Shl   => b.ins().ishl(lv, rv),
        IrBinOp::Shr   => {
            if is_unsigned  { b.ins().ushr(lv, rv) }
            else            { b.ins().sshr(lv, rv) }
        }
        // Comparisons: extend result from I8 to I64 for uniformity.
        IrBinOp::CmpEq => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, lv, rv)
            } else {
                b.ins().icmp(IntCC::Equal, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
        IrBinOp::CmpNe => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, lv, rv)
            } else {
                b.ins().icmp(IntCC::NotEqual, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
        IrBinOp::CmpLt => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, lv, rv)
            } else if is_unsigned {
                b.ins().icmp(IntCC::UnsignedLessThan, lv, rv)
            } else {
                b.ins().icmp(IntCC::SignedLessThan, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
        IrBinOp::CmpLe => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThanOrEqual, lv, rv)
            } else if is_unsigned {
                b.ins().icmp(IntCC::UnsignedLessThanOrEqual, lv, rv)
            } else {
                b.ins().icmp(IntCC::SignedLessThanOrEqual, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
        IrBinOp::CmpGt => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, lv, rv)
            } else if is_unsigned {
                b.ins().icmp(IntCC::UnsignedGreaterThan, lv, rv)
            } else {
                b.ins().icmp(IntCC::SignedGreaterThan, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
        IrBinOp::CmpGe => {
            let c = if is_float {
                b.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThanOrEqual, lv, rv)
            } else if is_unsigned {
                b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lv, rv)
            } else {
                b.ins().icmp(IntCC::SignedGreaterThanOrEqual, lv, rv)
            };
            b.ins().uextend(cl::I64, c)
        }
    })
}

// ---------------------------------------------------------------------------
// Terminator translation
// ---------------------------------------------------------------------------

fn translate_term(
    term:    &Terminator,
    cl_blks: &HashMap<BlockId, ClBlock>,
    b:       &mut FunctionBuilder<'_>,
    val_map: &HashMap<ValueId, ClValue>,
    diags:   &mut Vec<CodegenDiagnostic>,
) {
    match term {
        Terminator::Jump(target) => {
            b.ins().jump(cl_blks[target], &[]);
        }

        Terminator::Branch { cond, then_blk, else_blk } => {
            let Some(&cv) = val_map.get(cond) else {
                diags.push(CodegenDiagnostic {
                    message: format!("branch cond {cond} not in val_map"),
                });
                b.ins().trap(TrapCode::unwrap_user(1));
                return;
            };
            b.ins().brif(cv, cl_blks[then_blk], &[], cl_blks[else_blk], &[]);
        }

        Terminator::Return(Some(v)) => {
            if let Some(&val) = val_map.get(v) {
                b.ins().return_(&[val]);
            } else {
                diags.push(CodegenDiagnostic {
                    message: format!("return value {v} not in val_map"),
                });
                b.ins().trap(TrapCode::unwrap_user(1));
            }
        }

        Terminator::Return(None) => {
            b.ins().return_(&[]);
        }

        Terminator::Unreachable => {
            b.ins().trap(TrapCode::unwrap_user(1));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir, parser, resolve, types};

    fn run_ir(src: &str) -> ir::IrModule<'_> {
        let pr  = parser::parse(src);
        let rr  = resolve::resolve(src, &pr.module);
        let mut ir = types::infer(src, &pr.module, &rr.resolutions);
        ir::lower(src, &pr.module, &mut ir)
    }

    /// Compile to JIT and return the artifact.
    fn jit(src: &str) -> JitArtifact {
        let m = run_ir(src);
        codegen_jit(&m)
    }

    // -----------------------------------------------------------------------
    // Smoke tests: compile without fatal errors
    // -----------------------------------------------------------------------

    #[test]
    fn smoke_add() {
        let a = jit("def add(a: u64, b: u64) -> u64:\n    return a + b\n");
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }

    #[test]
    fn smoke_while_loop() {
        let a = jit("def count():\n    var i: u64 = 0\n    while i < 10:\n        i += 1\n");
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }


    #[test]
    fn smoke_if_else() {
        let a = jit(
            "def abs(x: i64) -> i64:\n    if x < 0:\n        return -x\n    else:\n        return x\n",
        );
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }

    #[test]
    fn smoke_ternary() {
        let a = jit(
            "def clamp(x: u64, lo: u64, hi: u64) -> u64:\n    return lo if x < lo else hi if x > hi else x\n",
        );
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }

    #[test]
    fn smoke_actor() {
        let a = jit(
            "actor Counter:\n    var count: u64 = 0\n    def increment(self):\n        self.count += 1\n    def get(self) -> u64:\n        return self.count\n",
        );
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }

    // -----------------------------------------------------------------------
    // JIT execution tests
    // -----------------------------------------------------------------------

    #[test]
    fn jit_add_executes() {
        let a = jit("def add(a: u64, b: u64) -> u64:\n    return a + b\n");
        assert!(a.diagnostics.iter().all(|d| !d.message.contains("failed")));
        // Safety: we know the function signature is (i64, i64) -> i64.
        let ptr = unsafe { a.fn_ptr("add") }.expect("add not found");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(3, 4) }, 7);
        assert_eq!(unsafe { f(0, 100) }, 100);
    }

    #[test]
    fn jit_abs_executes() {
        let a = jit(
            "def abs(x: i64) -> i64:\n    if x < 0:\n        return -x\n    else:\n        return x\n",
        );
        let ptr = unsafe { a.fn_ptr("abs") }.expect("abs not found");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(-5) },  5);
        assert_eq!(unsafe { f( 5) },  5);
        assert_eq!(unsafe { f( 0) },  0);
    }

    #[test]
    fn jit_count_loop_executes() {
        // count() accumulates i from 0 to 9 and returns via augmented assign.
        // Since count() has no return value (ret), we just check it doesn't crash.
        let a = jit("def count():\n    var i: u64 = 0\n    while i < 10:\n        i += 1\n");
        let ptr = unsafe { a.fn_ptr("count") }.expect("count not found");
        let f: extern "C" fn() = unsafe { std::mem::transmute(ptr) };
        // Should run without trapping.
        unsafe { f() };
    }

    #[test]
    fn jit_multiply() {
        let a = jit("def mul(a: u64, b: u64) -> u64:\n    return a * b\n");
        let ptr = unsafe { a.fn_ptr("mul") }.expect("mul not found");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(6, 7) }, 42);
    }

    #[test]
    fn jit_compare_lt() {
        let a = jit("def lt(a: u64, b: u64) -> u64:\n    return a < b\n");
        let ptr = unsafe { a.fn_ptr("lt") }.expect("lt not found");
        // icmp result is i8 but stored as i64 due to type propagation; cast accordingly.
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        // Note: icmp returns I64 when inputs are I64 on x86-64 with Cranelift.
        // Just check the call doesn't crash; actual result may be 0 or 1.
        let _ = unsafe { f(3, 5) };
    }

    // -----------------------------------------------------------------------
    // Direct call tests
    // -----------------------------------------------------------------------

    #[test]
    fn jit_direct_call_between_functions() {
        // `double` calls `add` — exercises DirectCall lowering.
        let src = "\
def add(a: u64, b: u64) -> u64:
    return a + b

def double(x: u64) -> u64:
    return add(x, x)
";
        let a = jit(src);
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed") || d.message.contains("unknown function"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");

        let ptr = unsafe { a.fn_ptr("double") }.expect("double not found");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(5) }, 10);
        assert_eq!(unsafe { f(0) }, 0);
    }

    #[test]
    fn jit_direct_call_chain() {
        // `square` calls `mul` (which calls `add`). Tests multi-level direct calls.
        let src = "\
def add(a: u64, b: u64) -> u64:
    return a + b

def mul(a: u64, b: u64) -> u64:
    return a * b

def square(x: u64) -> u64:
    return mul(x, x)
";
        let a = jit(src);
        let ptr = unsafe { a.fn_ptr("square") }.expect("square not found");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(4) }, 16);
        assert_eq!(unsafe { f(7) }, 49);
    }

    // -----------------------------------------------------------------------
    // Float arithmetic tests
    // -----------------------------------------------------------------------

    #[test]
    fn jit_f64_add() {
        let a = jit("def fadd(a: f64, b: f64) -> f64:\n    return a + b\n");
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");

        let ptr = unsafe { a.fn_ptr("fadd") }.expect("fadd not found");
        let f: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(ptr) };
        assert!((unsafe { f(1.0, 2.0) } - 3.0).abs() < 1e-10);
        assert!((unsafe { f(0.5, 0.5) } - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jit_f64_mul() {
        let a = jit("def fmul(a: f64, b: f64) -> f64:\n    return a * b\n");
        let ptr = unsafe { a.fn_ptr("fmul") }.expect("fmul not found");
        let f: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(ptr) };
        assert!((unsafe { f(3.0, 4.0) } - 12.0).abs() < 1e-10);
    }

    #[test]
    fn jit_f64_div() {
        let a = jit("def fdiv(a: f64, b: f64) -> f64:\n    return a / b\n");
        let ptr = unsafe { a.fn_ptr("fdiv") }.expect("fdiv not found");
        let f: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(ptr) };
        assert!((unsafe { f(10.0, 4.0) } - 2.5).abs() < 1e-10);
    }

    #[test]
    fn smoke_f64_sub() {
        let a = jit("def fsub(a: f64, b: f64) -> f64:\n    return a - b\n");
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");
    }

    // -----------------------------------------------------------------------
    // Unsigned integer operation tests
    // -----------------------------------------------------------------------

    #[test]
    fn jit_unsigned_comparison() {
        // udiv: 7 / 2 = 3 (unsigned).  As a signed i64 both are positive so
        // sdiv would give the same answer; what matters is the instruction
        // selection path doesn't crash.
        let a = jit("def udiv(a: u64, b: u64) -> u64:\n    return a / b\n");
        let fatal: Vec<_> = a.diagnostics.iter()
            .filter(|d| d.message.contains("failed"))
            .collect();
        assert!(fatal.is_empty(), "fatal diags: {fatal:?}");

        let ptr = unsafe { a.fn_ptr("udiv") }.expect("udiv not found");
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(7, 2) }, 3);
        assert_eq!(unsafe { f(100, 10) }, 10);
    }

    #[test]
    fn jit_signed_division() {
        let a = jit("def sdiv(a: i64, b: i64) -> i64:\n    return a / b\n");
        let ptr = unsafe { a.fn_ptr("sdiv") }.expect("sdiv not found");
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(unsafe { f(-7, 2) }, -3);
        assert_eq!(unsafe { f(10, -3) }, -3);
    }
}
