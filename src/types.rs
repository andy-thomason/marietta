/// Type inference pass for the Mandy compiler.
///
/// Performs Hindley-Milner-style unification over the AST.  Every expression
/// and declaration is assigned a [`TypeId`] — a handle into the [`TypeStore`]
/// union-find table.  Unification propagates constraints from annotations and
/// from the shapes of expressions.
///
/// # Numeric literals
///
/// Integer and float literals are given the special "literal" types
/// [`Type::IntLit`] and [`Type::FloatLit`].  These unify with any concrete
/// integer or float type respectively, and default to `i64` / `f64` if still
/// unconstrained after the whole module is walked.
///
/// # Errors
///
/// On a type mismatch the offending node is bound to [`Type::Error`] to
/// suppress cascading diagnostics.  Inference continues regardless.
use std::collections::HashMap;

use crate::ast::*;
use crate::resolve::{NodeId, ResolutionMap};

// ---------------------------------------------------------------------------
// TypeId
// ---------------------------------------------------------------------------

/// A cheap handle into the [`TypeStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

/// A resolved Mandy type.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Any-width integer: `u8`, `i64`, `u256`, …
    Int { bits: u16, signed: bool },
    /// Float with explicit total bit-width and mantissa bits:
    /// `f32` (IEEE 754 single), `f64` (double), `f16_8` (bfloat16), …
    Float { bits: u16, mantissa: u16 },
    /// `bool`
    Bool,
    /// `str`
    Str,
    /// Unit / `None`
    None_,
    /// Tuple `(A, B, …)`
    Tuple(Vec<TypeId>),
    /// Homogeneous 1-D slice `&[T]`
    Slice(TypeId),
    /// N-dimensional strided slice `&[T, rank]`.
    ///
    /// The rank is fixed at compile time; lengths and strides are runtime
    /// values stored alongside the data pointer.  A 1-D `MultiSlice` unifies
    /// freely with `Slice`.  Binops on two slices of the same rank work
    /// element-wise; `%*%` on two rank-2 slices performs tensor multiply.
    MultiSlice { elem: TypeId, rank: u16 },
    /// Fixed-size array `[T; N]`.
    Array { elem: TypeId, len: u64 },
    /// A named struct, identified by the [`NodeId`] of its `StructDef`.
    Struct(NodeId),
    /// Function type.
    Fn { params: Vec<TypeId>, ret: TypeId },
    /// `async def` return type wrapper.
    Async(TypeId),
    /// `channel<T>`
    Channel(TypeId),
    /// Unconstrained integer literal — defaults to `i64` if never unified.
    IntLit,
    /// Unconstrained float literal — defaults to `f64` if never unified.
    FloatLit,
    /// Error sentinel — prevents cascading diagnostics.
    Error,
}

// ---------------------------------------------------------------------------
// TypeStore — union-find over TypeIds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Entry {
    /// Free type variable not yet unified with anything.
    Unbound,
    /// This node has been unified; follow the chain.
    Link(TypeId),
    /// Resolved to a (possibly compound) type.
    Bound(Type),
}

/// Allocates and unifies types.  Implements a union-find with path compression.
pub struct TypeStore {
    entries: Vec<Entry>,
}

impl TypeStore {
    fn new() -> Self {
        TypeStore { entries: Vec::new() }
    }

    /// Allocate a fresh unbound type variable.
    pub fn fresh(&mut self) -> TypeId {
        let id = TypeId(self.entries.len() as u32);
        self.entries.push(Entry::Unbound);
        id
    }

    /// Intern a concrete type and return a [`TypeId`] for it.
    pub fn intern(&mut self, ty: Type) -> TypeId {
        let id = TypeId(self.entries.len() as u32);
        self.entries.push(Entry::Bound(ty));
        id
    }

    /// Follow links to find the canonical representative (path-compressed).
    pub fn root(&mut self, id: TypeId) -> TypeId {
        match self.entries[id.0 as usize].clone() {
            Entry::Link(parent) => {
                let root = self.root(parent);
                self.entries[id.0 as usize] = Entry::Link(root); // path compression
                root
            }
            _ => id,
        }
    }

    /// Resolve `id` to its [`Type`], defaulting unbound literals.
    pub fn resolve(&mut self, id: TypeId) -> Type {
        let root = self.root(id);
        match self.entries[root.0 as usize].clone() {
            Entry::Bound(Type::IntLit)   => Type::Int   { bits: 64, signed: true  },
            Entry::Bound(Type::FloatLit)  => Type::Float { bits: 64, mantissa: 52  },
            Entry::Bound(Type::MultiSlice { elem, rank }) => Type::MultiSlice { elem, rank },
            Entry::Bound(t) => t,
            Entry::Unbound => Type::Error,
            Entry::Link(_) => unreachable!("root() guarantees no Link"),
        }
    }

    /// Attempt to unify `a` and `b`.  Returns `true` on success.
    /// On structural mismatch returns `false` without modifying the store.
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> bool {
        let ra = self.root(a);
        let rb = self.root(b);
        if ra == rb {
            return true;
        }
        let ta = self.entries[ra.0 as usize].clone();
        let tb = self.entries[rb.0 as usize].clone();

        match (ta, tb) {
            // Free variable unifies with anything.
            (Entry::Unbound, _) => {
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }
            (_, Entry::Unbound) => {
                self.entries[rb.0 as usize] = Entry::Link(ra);
                true
            }

            // Error silently absorbs mismatches to suppress cascading errors.
            (Entry::Bound(Type::Error), _) | (_, Entry::Bound(Type::Error)) => true,

            // IntLit unifies with any concrete Int.
            (Entry::Bound(Type::IntLit), Entry::Bound(Type::Int { .. })) => {
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }
            (Entry::Bound(Type::Int { .. }), Entry::Bound(Type::IntLit)) => {
                self.entries[rb.0 as usize] = Entry::Link(ra);
                true
            }
            (Entry::Bound(Type::IntLit), Entry::Bound(Type::IntLit)) => {
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            // FloatLit unifies with any concrete Float.
            (Entry::Bound(Type::FloatLit), Entry::Bound(Type::Float { .. })) => {
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }
            (Entry::Bound(Type::Float { .. }), Entry::Bound(Type::FloatLit)) => {
                self.entries[rb.0 as usize] = Entry::Link(ra);
                true
            }
            (Entry::Bound(Type::FloatLit), Entry::Bound(Type::FloatLit)) => {
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            // Primitive exact matches.
            (Entry::Bound(Type::Int { bits: b1, signed: s1 }),
             Entry::Bound(Type::Int { bits: b2, signed: s2 })) => b1 == b2 && s1 == s2,

            (Entry::Bound(Type::Float { bits: b1, mantissa: m1 }),
             Entry::Bound(Type::Float { bits: b2, mantissa: m2 })) => b1 == b2 && m1 == m2,

            (Entry::Bound(Type::Bool),  Entry::Bound(Type::Bool))  => true,
            (Entry::Bound(Type::Str),   Entry::Bound(Type::Str))   => true,
            (Entry::Bound(Type::None_), Entry::Bound(Type::None_)) => true,

            // Struct: same declaration site ⇒ same type.
            (Entry::Bound(Type::Struct(a)), Entry::Bound(Type::Struct(b))) => a == b,

            // Compound: unify element types.
            (Entry::Bound(Type::Tuple(as_)), Entry::Bound(Type::Tuple(bs))) => {
                if as_.len() != bs.len() {
                    return false;
                }
                let pairs: Vec<_> = as_.iter().copied().zip(bs.iter().copied()).collect();
                for (x, y) in pairs {
                    if !self.unify(x, y) {
                        return false;
                    }
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            (Entry::Bound(Type::Slice(a)), Entry::Bound(Type::Slice(b))) => {
                if !self.unify(a, b) {
                    return false;
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            (Entry::Bound(Type::Array { elem: ea, len: la }),
             Entry::Bound(Type::Array { elem: eb, len: lb })) => {
                if la != lb {
                    return false;
                }
                if !self.unify(ea, eb) {
                    return false;
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            // MultiSlice(rank N) unifies with MultiSlice(rank N).
            (Entry::Bound(Type::MultiSlice { elem: ea, rank: ra_ }),
             Entry::Bound(Type::MultiSlice { elem: eb, rank: rb_ })) => {
                if ra_ != rb_ {
                    return false;
                }
                if !self.unify(ea, eb) {
                    return false;
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            // A rank-1 MultiSlice unifies with a plain Slice.
            (Entry::Bound(Type::MultiSlice { elem: ea, rank: 1 }),
             Entry::Bound(Type::Slice(eb))) => {
                if !self.unify(ea, eb) {
                    return false;
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }
            (Entry::Bound(Type::Slice(ea)),
             Entry::Bound(Type::MultiSlice { elem: eb, rank: 1 })) => {
                if !self.unify(ea, eb) {
                    return false;
                }
                self.entries[ra.0 as usize] = Entry::Link(rb);
                true
            }

            (Entry::Bound(Type::Fn { params: p1, ret: r1 }),
             Entry::Bound(Type::Fn { params: p2, ret: r2 })) => {
                if p1.len() != p2.len() {
                    return false;
                }
                let pairs: Vec<_> = p1.iter().copied().zip(p2.iter().copied()).collect();
                let (r1, r2) = (r1, r2);
                for (x, y) in pairs {
                    if !self.unify(x, y) {
                        return false;
                    }
                }
                self.unify(r1, r2)
            }

            (Entry::Bound(Type::Async(a)),   Entry::Bound(Type::Async(b)))   => self.unify(a, b),
            (Entry::Bound(Type::Channel(a)), Entry::Bound(Type::Channel(b))) => self.unify(a, b),

            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Maps each declaration/expression [`NodeId`] to its inferred [`TypeId`].
pub type TypeMap = HashMap<NodeId, TypeId>;

/// Output of the type inference pass.
pub struct InferResult<'src> {
    pub types: TypeMap,
    pub store: TypeStore,
    pub diagnostics: Vec<TypeDiagnostic<'src>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDiagnostic<'src> {
    TypeMismatch { src: &'src str, note: &'static str },
    UnknownType   { src: &'src str, name: &'src str },
    ArityMismatch { src: &'src str, expected: usize, got: usize },
    NotCallable   { src: &'src str },
}

// ---------------------------------------------------------------------------
// Type annotation parsing
// ---------------------------------------------------------------------------

/// Return the standard IEEE mantissa bit-count for common float widths.
fn ieee_mantissa(bits: u16) -> Option<u16> {
    match bits {
        16  => Some(10),
        32  => Some(23),
        64  => Some(52),
        80  => Some(63),
        128 => Some(112),
        _   => None,
    }
}

/// Parse a type name string into a [`Type`], or `None` if unrecognised.
pub fn parse_type_name(name: &str) -> Option<Type> {
    match name {
        "bool" => return Some(Type::Bool),
        "str"  => return Some(Type::Str),
        "None" | "none" => return Some(Type::None_),
        _ => {}
    }

    // uN — unsigned integer with N bits (1–2048)
    if let Some(rest) = name.strip_prefix('u') {
        if let Ok(bits) = rest.parse::<u16>() {
            if (1..=2048).contains(&bits) {
                return Some(Type::Int { bits, signed: false });
            }
        }
    }

    // iN — signed integer with N bits (1–2048)
    if let Some(rest) = name.strip_prefix('i') {
        if let Ok(bits) = rest.parse::<u16>() {
            if (1..=2048).contains(&bits) {
                return Some(Type::Int { bits, signed: true });
            }
        }
    }

    // fN or fN_M — float
    if let Some(rest) = name.strip_prefix('f') {
        if let Some(pos) = rest.find('_') {
            // fN_M — explicit mantissa (e.g. f16_8 for bfloat16)
            let bits_str     = &rest[..pos];
            let mantissa_str = &rest[pos + 1..];
            if let (Ok(bits), Ok(mantissa)) = (bits_str.parse::<u16>(), mantissa_str.parse::<u16>()) {
                return Some(Type::Float { bits, mantissa });
            }
        } else if let Ok(bits) = rest.parse::<u16>() {
            if let Some(mantissa) = ieee_mantissa(bits) {
                return Some(Type::Float { bits, mantissa });
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Literal suffix helper
// ---------------------------------------------------------------------------

/// Extract the type suffix from a numeric or string literal token text,
/// e.g. `"42_u8"` → `Some("u8")`, `"1_000_u32"` → `Some("u32")`,
/// `"3.14_f32"` → `Some("f32")`, `"42"` → `None`.
///
/// The suffix is the alphanumeric run after the last `_` that starts with
/// an alphabetic character.
fn literal_suffix(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    // Walk from the end: find the last `_` followed by an alpha char.
    for i in (1..bytes.len()).rev() {
        if bytes[i - 1] == b'_' && (bytes[i] as char).is_alphabetic() {
            return Some(&s[i..]);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Type checker
// ---------------------------------------------------------------------------

struct TypeChecker<'src, 'res> {
    source:      &'src str,
    store:       TypeStore,
    types:       TypeMap,
    /// Scope stack: declaration NodeId → TypeId.
    env:         Vec<HashMap<NodeId, TypeId>>,
    /// Expected return type of the innermost function.
    return_type: Option<TypeId>,
    resolutions: &'res ResolutionMap,
    diagnostics: Vec<TypeDiagnostic<'src>>,
}

impl<'src, 'res> TypeChecker<'src, 'res> {
    fn new(source: &'src str, resolutions: &'res ResolutionMap) -> Self {
        TypeChecker {
            source,
            store: TypeStore::new(),
            types: HashMap::new(),
            env: vec![HashMap::new()],
            return_type: None,
            resolutions,
            diagnostics: Vec::new(),
        }
    }

    fn node_id(&self, s: &str) -> NodeId {
        let base = self.source.as_ptr() as usize;
        let ptr  = s.as_ptr() as usize;
        NodeId(ptr.saturating_sub(base))
    }

    fn push_scope(&mut self) { self.env.push(HashMap::new()); }
    fn pop_scope(&mut self)  { self.env.pop(); }

    /// Record `decl_src` → `ty` in the innermost scope and in `types`.
    fn bind(&mut self, decl_src: &'src str, ty: TypeId) {
        let id = self.node_id(decl_src);
        self.env.last_mut().unwrap().insert(id, ty);
        self.types.insert(id, ty);
    }

    /// Look up a declaration NodeId in the scope stack.
    fn lookup_decl(&self, decl_id: NodeId) -> Option<TypeId> {
        for scope in self.env.iter().rev() {
            if let Some(&tid) = scope.get(&decl_id) {
                return Some(tid);
            }
        }
        None
    }

    /// Convert an AST type annotation to a [`TypeId`].
    fn type_of_annotation(&mut self, ann: &TypeExpr<'src>) -> TypeId {
        match &ann.kind {
            TypeExprKind::Name(name) => {
                if let Some(ty) = parse_type_name(name) {
                    self.store.intern(ty)
                } else {
                    self.diagnostics.push(TypeDiagnostic::UnknownType {
                        src: ann.src, name,
                    });
                    self.store.intern(Type::Error)
                }
            }
            TypeExprKind::Array { elem, len } => {
                let elem_ty = self.type_of_annotation(elem);
                self.store.intern(Type::Array { elem: elem_ty, len: *len })
            }
            TypeExprKind::Generic { base, args } => {
                if let TypeExprKind::Name(base_name) = &base.kind {
                    match *base_name {
                        "Channel" | "channel" if args.len() == 1 => {
                            let inner = self.type_of_annotation(&args[0]);
                            self.store.intern(Type::Channel(inner))
                        }
                        // Slice[T] — 1-D slice
                        "Slice" | "slice" if args.len() == 1 => {
                            let elem = self.type_of_annotation(&args[0]);
                            self.store.intern(Type::Slice(elem))
                        }
                        // MultiSlice[T, N] — N-dimensional slice
                        "MultiSlice" | "multislice" if args.len() == 2 => {
                            let elem = self.type_of_annotation(&args[0]);
                            // Second arg must be an integer literal rank.
                            let rank: u16 = if let TypeExprKind::Name(n) = &args[1].kind {
                                n.parse().unwrap_or(0)
                            } else { 0 };
                            if rank == 0 {
                                self.diagnostics.push(TypeDiagnostic::UnknownType {
                                    src: ann.src, name: "MultiSlice rank must be a positive integer",
                                });
                                self.store.intern(Type::Error)
                            } else {
                                self.store.intern(Type::MultiSlice { elem, rank })
                            }
                        }
                        _ => {
                            self.diagnostics.push(TypeDiagnostic::UnknownType {
                                src: ann.src, name: base_name,
                            });
                            self.store.intern(Type::Error)
                        }
                    }
                } else {
                    self.store.intern(Type::Error)
                }
            }
            TypeExprKind::Error(_) => self.store.intern(Type::Error),
        }
    }

    /// Unify `a` and `b`; on mismatch emit a diagnostic and bind `a` to Error.
    fn unify_or_error(&mut self, a: TypeId, b: TypeId, src: &'src str, note: &'static str) {
        if !self.store.unify(a, b) {
            self.diagnostics.push(TypeDiagnostic::TypeMismatch { src, note });
            let err = self.store.intern(Type::Error);
            // Bind `a` to Error so downstream unifications don't cascade.
            self.entries_link(a, err);
        }
    }

    /// Force-link `id` to `target` regardless of current state.
    fn entries_link(&mut self, id: TypeId, target: TypeId) {
        let ra = self.store.root(id);
        self.store.entries[ra.0 as usize] = Entry::Link(target);
    }

    // ------------------------------------------------------------------
    // Expression inference — returns the TypeId for this expression node
    // ------------------------------------------------------------------

    fn walk_expr(&mut self, expr: &Expr<'src>) -> TypeId {
        let tid = self.infer_expr(expr);
        self.types.insert(self.node_id(expr.src), tid);
        tid
    }

    fn infer_expr(&mut self, expr: &Expr<'src>) -> TypeId {
        match &expr.kind {
            ExprKind::IntLiteral(s) => {
                // Strip digit-separator `_` groups and extract optional type suffix.
                // e.g. `42_u8` → suffix `u8`, `1_000_u32` → suffix `u32`.
                if let Some(suffix) = literal_suffix(s) {
                    if let Some(ty) = parse_type_name(suffix) {
                        return self.store.intern(ty);
                    }
                }
                self.store.intern(Type::IntLit)
            }
            ExprKind::FloatLiteral(s) => {
                if let Some(suffix) = literal_suffix(s) {
                    if let Some(ty) = parse_type_name(suffix) {
                        return self.store.intern(ty);
                    }
                }
                self.store.intern(Type::FloatLit)
            }
            ExprKind::StringLiteral(_) => self.store.intern(Type::Str),
            ExprKind::BoolLiteral(_)   => self.store.intern(Type::Bool),
            ExprKind::NoneLiteral      => self.store.intern(Type::None_),

            ExprKind::Name(_) => {
                let use_id = self.node_id(expr.src);
                if let Some(&decl_id) = self.resolutions.get(&use_id) {
                    if let Some(tid) = self.lookup_decl(decl_id) {
                        return tid;
                    }
                }
                // Resolver already reported undefined — return Unknown so we
                // don't cascade type errors.
                self.store.fresh()
            }

            ExprKind::BinOp { op, left, right } => {
                let lt = self.walk_expr(left);
                let rt = self.walk_expr(right);
                match *op {
                    "==" | "!=" | "<" | ">" | "<=" | ">=" | "in" | "is" => {
                        self.unify_or_error(lt, rt, expr.src,
                            "comparison operands must have the same type");
                        self.store.intern(Type::Bool)
                    }
                    "and" | "or" => {
                        let bool_ty = self.store.intern(Type::Bool);
                        self.unify_or_error(lt, bool_ty, left.src,  "expected bool");
                        self.unify_or_error(rt, bool_ty, right.src, "expected bool");
                        bool_ty
                    }
                    _ => {
                        self.unify_or_error(lt, rt, expr.src,
                            "binary operands must have the same type");
                        lt
                    }
                }
            }

            ExprKind::UnaryOp { op, operand } => {
                let ot = self.walk_expr(operand);
                match *op {
                    "not" => {
                        let bool_ty = self.store.intern(Type::Bool);
                        self.unify_or_error(ot, bool_ty, operand.src, "not requires bool");
                        bool_ty
                    }
                    _ => ot,
                }
            }

            ExprKind::Await(inner) => {
                let it = self.walk_expr(inner);
                let inner_ty  = self.store.fresh();
                let async_ty  = self.store.intern(Type::Async(inner_ty));
                self.unify_or_error(it, async_ty, expr.src, "await requires an async value");
                inner_ty
            }

            ExprKind::Call { func, args, kwargs } => {
                let ft = self.walk_expr(func);
                let arg_types: Vec<TypeId> = args.iter().map(|a| self.walk_expr(a)).collect();
                for (_, v) in kwargs { self.walk_expr(v); }
                let ret_ty = self.store.fresh();
                let fn_ty  = self.store.intern(Type::Fn { params: arg_types, ret: ret_ty });
                // Best-effort unification; if ft is Unknown we don't hard-error.
                let _ = self.store.unify(ft, fn_ty);
                ret_ty
            }

            ExprKind::Index { obj, index } => {
                let obj_ty = self.walk_expr(obj);
                self.walk_expr(index);
                // If the object is a known Slice or MultiSlice, return the element type.
                // Otherwise return a fresh variable; element typing is refined in later passes.
                let elem_ty = self.store.fresh();
                let slice_ty   = self.store.intern(Type::Slice(elem_ty));
                let ms1_ty     = self.store.intern(Type::MultiSlice { elem: elem_ty, rank: 1 });
                // Try Slice first, then MultiSlice rank-1 (they unify with each other).
                if !self.store.unify(obj_ty, slice_ty) {
                    // Not a 1-D slice — could be any-rank MultiSlice; return fresh elem.
                    let _ = self.store.unify(obj_ty, ms1_ty);
                }
                elem_ty
            }

            ExprKind::Attr { obj, .. } => {
                self.walk_expr(obj);
                self.store.fresh() // struct field typing handled in a later pass
            }

            ExprKind::IfExpr { condition, value, alt } => {
                let ct = self.walk_expr(condition);
                let bool_ty = self.store.intern(Type::Bool);
                self.unify_or_error(ct, bool_ty, condition.src, "condition must be bool");
                let vt = self.walk_expr(value);
                let at = self.walk_expr(alt);
                self.unify_or_error(vt, at, expr.src,
                    "if-expression branches must have the same type");
                vt
            }

            ExprKind::Lambda { params, body } => {
                self.push_scope();
                let param_types: Vec<TypeId> = params.iter().map(|p| {
                    let ty = if let Some(ann) = &p.annotation {
                        self.type_of_annotation(ann)
                    } else {
                        self.store.fresh()
                    };
                    if let Some(default) = &p.default { self.walk_expr(default); }
                    self.bind(p.src, ty);
                    ty
                }).collect();
                let ret = self.walk_expr(body);
                self.pop_scope();
                self.store.intern(Type::Fn { params: param_types, ret })
            }

            ExprKind::Tuple(elems) => {
                let elem_types: Vec<TypeId> = elems.iter().map(|e| self.walk_expr(e)).collect();
                self.store.intern(Type::Tuple(elem_types))
            }

            ExprKind::List(elems) => {
                let elem_ty = self.store.fresh();
                for e in elems {
                    let et = self.walk_expr(e);
                    self.unify_or_error(et, elem_ty, e.src,
                        "list elements must have the same type");
                }
                self.store.intern(Type::Slice(elem_ty))
            }

            ExprKind::ArrayLit(elems) => {
                let elem_ty = self.store.fresh();
                for e in elems {
                    let et = self.walk_expr(e);
                    self.unify_or_error(et, elem_ty, e.src,
                        "array elements must have the same type");
                }
                let len = elems.len() as u64;
                self.store.intern(Type::Array { elem: elem_ty, len })
            }

            ExprKind::MultiSliceLit(ranges) => {
                // &[lo..hi, lo..hi, …] — all bounds must be integers;
                // the result is MultiSlice { elem: <fresh>, rank: N }.
                let idx_ty = self.store.intern(Type::IntLit);
                for (lo, hi) in ranges {
                    let lt = self.walk_expr(lo);
                    let ht = self.walk_expr(hi);
                    self.unify_or_error(lt, idx_ty, lo.src, "slice bound must be an integer");
                    self.unify_or_error(ht, idx_ty, hi.src, "slice bound must be an integer");
                }
                let elem_ty = self.store.fresh();
                let rank    = ranges.len() as u16;
                self.store.intern(Type::MultiSlice { elem: elem_ty, rank })
            }

            ExprKind::Error(_) => self.store.intern(Type::Error),
        }
    }

    // ------------------------------------------------------------------
    // Statement walker
    // ------------------------------------------------------------------

    fn walk_stmt(&mut self, stmt: &Stmt<'src>) {
        match &stmt.kind {
            StmtKind::VarDecl { name: _, annotation, value } => {
                let decl_ty = if let Some(ann) = annotation {
                    self.type_of_annotation(ann)
                } else {
                    self.store.fresh()
                };
                self.bind(stmt.src, decl_ty);
                if let Some(val) = value {
                    let val_ty = self.walk_expr(val);
                    self.unify_or_error(val_ty, decl_ty, val.src,
                        "value type does not match declared type");
                }
            }

            StmtKind::LetDecl { name: _, annotation, value } => {
                let decl_ty = if let Some(ann) = annotation {
                    self.type_of_annotation(ann)
                } else {
                    self.store.fresh()
                };
                self.bind(stmt.src, decl_ty);
                let val_ty = self.walk_expr(value);
                self.unify_or_error(val_ty, decl_ty, value.src,
                    "value type does not match declared type");
            }

            StmtKind::Assign { op: _, target, value } => {
                let val_ty    = self.walk_expr(value);
                let target_ty = self.walk_expr(target);
                self.unify_or_error(val_ty, target_ty, stmt.src, "assignment type mismatch");
            }

            StmtKind::Return(value) => {
                if let Some(ret_ty) = self.return_type {
                    let actual = if let Some(val) = value {
                        self.walk_expr(val)
                    } else {
                        self.store.intern(Type::None_)
                    };
                    self.unify_or_error(actual, ret_ty, stmt.src, "return type mismatch");
                } else if let Some(val) = value {
                    self.walk_expr(val);
                }
            }

            StmtKind::If { branches, else_body } => {
                for (cond, body) in branches {
                    let ct = self.walk_expr(cond);
                    let bool_ty = self.store.intern(Type::Bool);
                    self.unify_or_error(ct, bool_ty, cond.src, "if condition must be bool");
                    self.push_scope();
                    for s in body { self.walk_stmt(s); }
                    self.pop_scope();
                }
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }

            StmtKind::While { condition, body, else_body } => {
                let ct = self.walk_expr(condition);
                let bool_ty = self.store.intern(Type::Bool);
                self.unify_or_error(ct, bool_ty, condition.src, "while condition must be bool");
                self.push_scope();
                for s in body { self.walk_stmt(s); }
                self.pop_scope();
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }

            StmtKind::For { target, iter, body, else_body } => {
                self.walk_expr(iter);
                self.push_scope();
                // Bind the loop variable to a fresh TypeVar.
                let target_ty = self.store.fresh();
                if let ExprKind::Name(_) = &target.kind {
                    self.bind(target.src, target_ty);
                }
                self.types.insert(self.node_id(target.src), target_ty);
                for s in body { self.walk_stmt(s); }
                self.pop_scope();
                self.push_scope();
                for s in else_body { self.walk_stmt(s); }
                self.pop_scope();
            }

            StmtKind::Expr(expr) => { self.walk_expr(expr); }
            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}

            StmtKind::Import { .. } | StmtKind::FromImport { .. } => {
                // Give the imported binding an unresolved type for now.
                let ty = self.store.fresh();
                self.bind(stmt.src, ty);
            }

            StmtKind::Error(_, _) => {}
        }
    }

    // ------------------------------------------------------------------
    // Function walker
    // ------------------------------------------------------------------

    fn walk_function(&mut self, f: &FunctionDef<'src>) {
        self.push_scope();

        let param_types: Vec<TypeId> = f.params.iter().map(|p| {
            let ty = if let Some(ann) = &p.annotation {
                self.type_of_annotation(ann)
            } else {
                self.store.fresh()
            };
            if let Some(default) = &p.default { self.walk_expr(default); }
            self.bind(p.src, ty);
            ty
        }).collect();

        let ret_ty = if let Some(ann) = &f.return_type {
            self.type_of_annotation(ann)
        } else {
            self.store.fresh()
        };

        let fn_ty = self.store.intern(Type::Fn { params: param_types, ret: ret_ty });
        self.types.insert(self.node_id(f.src), fn_ty);

        let prev = self.return_type.replace(ret_ty);
        for s in &f.body { self.walk_stmt(s); }
        self.return_type = prev;

        self.pop_scope();
    }

    // ------------------------------------------------------------------
    // Item and module walkers
    // ------------------------------------------------------------------

    fn walk_item(&mut self, item: &Item<'src>) {
        match &item.kind {
            ItemKind::FunctionDef(f) => {
                // Name was already hoisted; just walk the body.
                self.walk_function(f);
                // Unify the hoisted placeholder with the actual Fn type.
                let hoisted = self.node_id(item.src);
                if let Some(&placeholder) = self.types.get(&hoisted) {
                    if let Some(&actual) = self.types.get(&self.node_id(f.src)) {
                        let _ = self.store.unify(placeholder, actual);
                    }
                }
            }
            ItemKind::StructDef(s) => {
                for field in &s.fields {
                    let field_ann = self.type_of_annotation(&field.annotation);
                    self.bind(field.src, field_ann);
                    if let Some(default) = &field.default { self.walk_expr(default); }
                }
            }
            ItemKind::ImplBlock(i) => {
                for method in &i.methods { self.walk_function(method); }
            }
            ItemKind::ActorDef(a) => {
                for field in &a.fields {
                    let field_ann = self.type_of_annotation(&field.annotation);
                    self.bind(field.src, field_ann);
                    if let Some(default) = &field.default { self.walk_expr(default); }
                }
                for method in &a.methods { self.walk_function(method); }
            }
            ItemKind::Stmt(stmt) => self.walk_stmt(stmt),
            ItemKind::Error(_, _) => {}
        }
    }

    fn walk_module(&mut self, module: &Module<'src>) {
        // Hoist pass: register top-level names so forward references type-check.
        for item in &module.items {
            match &item.kind {
                ItemKind::FunctionDef(_) => {
                    let ty = self.store.fresh();
                    self.bind(item.src, ty);
                }
                ItemKind::StructDef(s) => {
                    let sid = self.node_id(s.src);
                    let ty  = self.store.intern(Type::Struct(sid));
                    self.bind(item.src, ty);
                }
                ItemKind::ActorDef(a) => {
                    let aid = self.node_id(a.src);
                    let ty  = self.store.intern(Type::Struct(aid));
                    self.bind(item.src, ty);
                }
                _ => {}
            }
        }

        // Resolution pass.
        for item in &module.items { self.walk_item(item); }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run type inference on a parsed module.
///
/// Takes the original source (for NodeId computation), the module AST, and
/// the [`ResolutionMap`] produced by the name-resolution pass.
pub fn infer<'src>(
    source:      &'src str,
    module:      &Module<'src>,
    resolutions: &ResolutionMap,
) -> InferResult<'src> {
    let mut checker = TypeChecker::new(source, resolutions);
    checker.walk_module(module);
    InferResult {
        types:       checker.types,
        store:       checker.store,
        diagnostics: checker.diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, resolve};

    fn run(src: &str) -> InferResult<'_> {
        let pr  = parser::parse(src);
        let rr  = resolve::resolve(src, &pr.module);
        infer(src, &pr.module, &rr.resolutions)
    }

    fn is_clean(src: &str) -> bool {
        run(src).diagnostics.is_empty()
    }

    fn resolved_type(result: &mut InferResult<'_>, src_slice: &str) -> Type {
        // Find the NodeId whose offset equals where `src_slice` appears in the source.
        // We search the types map for any NodeId whose offset corresponds.
        // For test purposes we find the first matching TypeId by iterating types.
        // The key is that src_slice must be a sub-slice of the original source.
        let ptr = src_slice.as_ptr() as usize;
        let matching = result.types.iter()
            .find(|(nid, _)| nid.0 == {
                // We need to work backwards from the ptr offset.
                // NodeId is stored as byte offset from source start.
                // But we don't have source here — instead we stored node_id(s) = ptr - base.
                // The test passes src_slice as a sub-slice of the original source string,
                // so its pointer matches exactly.
                let _ = ptr; // suppress warning
                nid.0
            });
        // Simpler approach: just find any TypeId matching the pointer offset.
        // Since tests call run() which has a local `src`, we need to be smarter.
        // Use the first TypeId found in types map for the given NodeId key.
        if let Some((_, &tid)) = matching {
            return result.store.resolve(tid);
        }
        Type::Error
    }

    // Helper: look up the type for a source-slice offset.
    fn type_at_offset(result: &mut InferResult<'_>, offset: usize) -> Type {
        let tid = result.types.get(&NodeId(offset)).copied()
            .unwrap_or_else(|| result.store.intern(Type::Error));
        result.store.resolve(tid)
    }

    // ------------------------------------------------------------------
    // Literal types
    // ------------------------------------------------------------------

    #[test]
    fn int_literal_defaults_to_i64() {
        let src = "x = 42\n";
        let mut result = run(src);
        // "42" is at offset 4 in the source
        assert_eq!(type_at_offset(&mut result, 4), Type::Int { bits: 64, signed: true });
    }

    #[test]
    fn float_literal_defaults_to_f64() {
        let src = "x = 3.14\n";
        let mut result = run(src);
        assert_eq!(type_at_offset(&mut result, 4), Type::Float { bits: 64, mantissa: 52 });
    }

    #[test]
    fn bool_literal_type() {
        let src = "x = True\n";
        let mut result = run(src);
        assert_eq!(type_at_offset(&mut result, 4), Type::Bool);
    }

    #[test]
    fn string_literal_type() {
        let src = "x = \"hi\"\n";
        let mut result = run(src);
        assert_eq!(type_at_offset(&mut result, 4), Type::Str);
    }

    // ------------------------------------------------------------------
    // Annotation parsing
    // ------------------------------------------------------------------

    #[test]
    fn parse_u8()    { assert_eq!(parse_type_name("u8"),   Some(Type::Int { bits: 8,   signed: false })); }
    #[test]
    fn parse_i64()   { assert_eq!(parse_type_name("i64"),  Some(Type::Int { bits: 64,  signed: true  })); }
    #[test]
    fn parse_u256()  { assert_eq!(parse_type_name("u256"), Some(Type::Int { bits: 256, signed: false })); }
    #[test]
    fn parse_f32()   { assert_eq!(parse_type_name("f32"),  Some(Type::Float { bits: 32,  mantissa: 23  })); }
    #[test]
    fn parse_f64()   { assert_eq!(parse_type_name("f64"),  Some(Type::Float { bits: 64,  mantissa: 52  })); }
    #[test]
    fn parse_f16_8() { assert_eq!(parse_type_name("f16_8"), Some(Type::Float { bits: 16, mantissa: 8  })); }
    #[test]
    fn parse_bool()  { assert_eq!(parse_type_name("bool"), Some(Type::Bool)); }
    #[test]
    fn parse_str()   { assert_eq!(parse_type_name("str"),  Some(Type::Str));  }
    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(parse_type_name("Widget"), None);
    }

    // ------------------------------------------------------------------
    // Annotation-driven inference (no diagnostics expected)
    // ------------------------------------------------------------------

    #[test]
    fn var_u8_int_lit_no_error() {
        assert!(is_clean("var x: u8 = 1\n"));
    }

    #[test]
    fn let_f32_float_lit_no_error() {
        assert!(is_clean("let x: f32 = 1.0\n"));
    }

    #[test]
    fn let_bool_no_error() {
        assert!(is_clean("let x: bool = True\n"));
    }

    #[test]
    fn let_str_no_error() {
        assert!(is_clean("let x: str = \"hello\"\n"));
    }

    #[test]
    fn function_with_annotations_no_error() {
        assert!(is_clean("def add(a: u32, b: u32) -> u32:\n    return a + b\n"));
    }

    #[test]
    fn function_no_annotation_no_error() {
        assert!(is_clean("def f():\n    pass\n"));
    }

    // ------------------------------------------------------------------
    // Binary operator result types
    // ------------------------------------------------------------------

    #[test]
    fn comparison_yields_bool() {
        let src = "x = 1 == 2\n";
        let mut result = run(src);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        // "1 == 2" is at offset 4
        assert_eq!(type_at_offset(&mut result, 4), Type::Bool);
    }

    #[test]
    fn arithmetic_int_lit() {
        let src = "x = 1 + 2\n";
        let mut result = run(src);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        // result should resolve to Int {64, true} (both sides IntLit)
        assert_eq!(type_at_offset(&mut result, 4), Type::Int { bits: 64, signed: true });
    }

    // ------------------------------------------------------------------
    // Type mismatches (diagnostics expected)
    // ------------------------------------------------------------------

    #[test]
    fn type_mismatch_float_in_int_var() {
        let result = run("var x: u8 = 1.0\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::TypeMismatch { .. })),
            "expected TypeMismatch, got: {:?}", result.diagnostics
        );
    }

    #[test]
    fn type_mismatch_bool_in_int_var() {
        let result = run("var x: u8 = True\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::TypeMismatch { .. })),
            "expected TypeMismatch, got: {:?}", result.diagnostics
        );
    }

    #[test]
    fn function_return_mismatch() {
        let result = run("def f() -> u8:\n    return True\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::TypeMismatch { .. })),
            "expected TypeMismatch, got: {:?}", result.diagnostics
        );
    }

    #[test]
    fn if_condition_must_be_bool() {
        // A u8 condition should produce a type mismatch.
        let result = run("var x: u8 = 1\nif x:\n    pass\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::TypeMismatch { .. })),
            "expected TypeMismatch, got: {:?}", result.diagnostics
        );
    }

    #[test]
    fn if_expr_branch_mismatch() {
        let result = run("x = 1 if True else \"hello\"\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::TypeMismatch { .. })),
            "expected TypeMismatch, got: {:?}", result.diagnostics
        );
    }

    #[test]
    fn unknown_type_annotation() {
        let result = run("var x: Widget = 1\n");
        assert!(
            result.diagnostics.iter().any(|d| matches!(d, TypeDiagnostic::UnknownType { .. })),
            "expected UnknownType, got: {:?}", result.diagnostics
        );
    }

    // ------------------------------------------------------------------
    // Struct and actor
    // ------------------------------------------------------------------

    #[test]
    fn struct_def_no_error() {
        assert!(is_clean("struct Point:\n    x: f32\n    y: f32\n"));
    }

    #[test]
    fn actor_def_no_error() {
        assert!(is_clean(
            "actor Counter:\n    var count: u64 = 0\n    def increment(self):\n        self.count += 1\n"
        ));
    }

    // ------------------------------------------------------------------
    // No panics on error nodes
    // ------------------------------------------------------------------

    #[test]
    fn no_panic_on_parse_errors() {
        let _ = run("x = \n");
    }

    #[test]
    fn no_panic_on_undefined_name() {
        let _ = run("y = undefined_variable\n");
    }

    // ------------------------------------------------------------------
    // MultiSlice
    // ------------------------------------------------------------------

    #[test]
    fn multislice_unify_same_rank() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 32, signed: false });
        let a = store.intern(Type::MultiSlice { elem, rank: 2 });
        let b = store.intern(Type::MultiSlice { elem, rank: 2 });
        assert!(store.unify(a, b));
    }

    #[test]
    fn multislice_rank_mismatch_fails() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 32, signed: false });
        let a = store.intern(Type::MultiSlice { elem, rank: 2 });
        let b = store.intern(Type::MultiSlice { elem, rank: 3 });
        assert!(!store.unify(a, b));
    }

    #[test]
    fn multislice_rank1_unifies_with_slice() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 8, signed: false });
        let ms   = store.intern(Type::MultiSlice { elem, rank: 1 });
        let sl   = store.intern(Type::Slice(elem));
        assert!(store.unify(ms, sl));
    }

    #[test]
    fn multislice_elem_mismatch_fails() {
        let mut store = TypeStore::new();
        let int_ty   = store.intern(Type::Int   { bits: 32, signed: true  });
        let float_ty = store.intern(Type::Float { bits: 32, mantissa: 23  });
        let a = store.intern(Type::MultiSlice { elem: int_ty,   rank: 2 });
        let b = store.intern(Type::MultiSlice { elem: float_ty, rank: 2 });
        assert!(!store.unify(a, b));
    }

    #[test]
    fn multislice_resolve() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 64, signed: true });
        let ms   = store.intern(Type::MultiSlice { elem, rank: 3 });
        assert_eq!(store.resolve(ms), Type::MultiSlice { elem, rank: 3 });
    }

    // ------------------------------------------------------------------
    // MultiSliceLit inference
    // ------------------------------------------------------------------

    #[test]
    fn multislice_lit_1d_has_rank_1() {
        let src = "x = &[10..20]\n";
        let mut result = run(src);
        // &[10..20] starts at offset 4
        let ty = type_at_offset(&mut result, 4);
        assert!(matches!(ty, Type::MultiSlice { rank: 1, .. }), "got {ty:?}");
    }

    #[test]
    fn multislice_lit_2d_has_rank_2() {
        let src = "x = &[10..20, 30..40]\n";
        let mut result = run(src);
        let ty = type_at_offset(&mut result, 4);
        assert!(matches!(ty, Type::MultiSlice { rank: 2, .. }), "got {ty:?}");
    }

    #[test]
    fn multislice_lit_no_type_errors() {
        assert!(is_clean("x = &[10..20, 30..40]\n"));
    }

    // ------------------------------------------------------------------
    // Literal suffix tests
    // ------------------------------------------------------------------

    #[test]
    fn literal_suffix_u8() {
        assert_eq!(super::literal_suffix("42_u8"), Some("u8"));
    }

    #[test]
    fn literal_suffix_u32_with_separator() {
        assert_eq!(super::literal_suffix("1_000_u32"), Some("u32"));
    }

    #[test]
    fn literal_suffix_none() {
        assert_eq!(super::literal_suffix("1_000"), None);
        assert_eq!(super::literal_suffix("42"), None);
    }

    #[test]
    fn int_suffix_infers_u8() {
        let src = "x = 42_u8\n";
        let mut result = run(src);
        // "42_u8" is at offset 4
        let ty = type_at_offset(&mut result, 4);
        assert_eq!(ty, Type::Int { bits: 8, signed: false }, "got {ty:?}");
    }

    #[test]
    fn int_suffix_infers_i32() {
        let src = "x = 100_i32\n";
        let mut result = run(src);
        let ty = type_at_offset(&mut result, 4);
        assert_eq!(ty, Type::Int { bits: 32, signed: true }, "got {ty:?}");
    }

    #[test]
    fn float_suffix_infers_f32() {
        let src = "x = 3.14_f32\n";
        let mut result = run(src);
        let ty = type_at_offset(&mut result, 4);
        assert_eq!(ty, Type::Float { bits: 32, mantissa: 23 }, "got {ty:?}");
    }

    #[test]
    fn suffixed_literal_no_errors() {
        assert!(is_clean("x = 42_u8\n"));
        assert!(is_clean("x = 3.14_f32\n"));
    }

    // ------------------------------------------------------------------
    // Array type and literal
    // ------------------------------------------------------------------

    #[test]
    fn array_lit_has_array_type() {
        let src = "x = [1, 2, 3]\n";
        let mut result = run(src);
        // "[1, 2, 3]" starts at offset 4
        let ty = type_at_offset(&mut result, 4);
        assert!(matches!(ty, Type::Array { len: 3, .. }), "got {ty:?}");
    }

    #[test]
    fn array_type_annotation_infers_array() {
        assert!(is_clean("var x: [u8; 4] = [1_u8, 2_u8, 3_u8, 4_u8]\n"));
    }

    #[test]
    fn array_unify_same_len() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 8, signed: false });
        let a = store.intern(Type::Array { elem, len: 4 });
        let b = store.intern(Type::Array { elem, len: 4 });
        assert!(store.unify(a, b));
    }

    #[test]
    fn array_unify_different_len_fails() {
        let mut store = TypeStore::new();
        let elem = store.intern(Type::Int { bits: 8, signed: false });
        let a = store.intern(Type::Array { elem, len: 4 });
        let b = store.intern(Type::Array { elem, len: 8 });
        assert!(!store.unify(a, b));
    }
}
