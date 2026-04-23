/// Actor model analysis pass for the Emmy compiler.
///
/// This pass walks every `actor` declaration in the module and produces an
/// [`ActorAnalysis`] for each one.  Code generation (step 9) uses these
/// descriptions to emit:
///
/// * A state struct holding the actor's persistent fields.
/// * A message enum with one variant per dispatchable method (methods whose
///   first parameter is `self`).
/// * A `run(self)` dispatch loop that receives messages from an MPSC channel
///   and calls the appropriate method, sending a reply when the method has a
///   non-void return type.
///
/// # Channel and RPC model
///
/// Every actor gets a pair of MPSC channels at spawn time:
/// * An **inbox** `Receiver<{Name}Msg>` the actor listens on.
/// * A **handle** `Sender<{Name}Msg>` given to callers.
///
/// For methods with a return type, the message variant carries a reply
/// `Sender<ReturnType>` so the caller can `await` the response.
///
/// # Limitations (v0)
///
/// * All methods are treated as public.  Access control is deferred.
/// * Methods without a `self` parameter are not dispatchable; a diagnostic is
///   emitted and the method is excluded from the message enum.
/// * Actor inheritance / delegation is not yet supported.

use crate::ast::*;

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// One parameter of a dispatchable method, excluding `self`.
#[derive(Debug, Clone, PartialEq)]
pub struct MsgParam<'src> {
    /// Parameter name.
    pub name: &'src str,
    /// Type annotation, if present in the source.
    pub annotation: Option<TypeExpr<'src>>,
}

/// One variant in the generated message enum — one per dispatchable method.
///
/// For a void method this is a plain tuple variant carrying only the method
/// arguments.  For a non-void method an implicit reply channel parameter is
/// also added during code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageVariant<'src> {
    /// Method name as it appears in source, e.g. `"increment"`.
    pub method_name: &'src str,
    /// PascalCase variant name, e.g. `"Increment"`.
    pub variant_name: String,
    /// Method parameters excluding `self`.
    pub params: Vec<MsgParam<'src>>,
    /// Return type of the method; `None` means void (fire-and-forget message).
    pub reply_type: Option<TypeExpr<'src>>,
    /// Whether the method is `async`.
    pub is_async: bool,
}

/// Complete analysis of one `actor` definition.
///
/// Code generation consumes this struct to emit the state struct, message
/// enum, and dispatch loop for the actor.
#[derive(Debug, Clone, PartialEq)]
pub struct ActorAnalysis<'src> {
    /// Source slice of the `actor` keyword (for diagnostics).
    pub src: &'src str,
    /// Name as declared, e.g. `"Counter"`.
    pub actor_name: &'src str,
    /// Name of the generated message enum, e.g. `"CounterMsg"`.
    pub msg_enum_name: String,
    /// Persistent state fields of the actor.
    pub fields: Vec<StructField<'src>>,
    /// One entry per dispatchable method (those with a leading `self` param).
    pub messages: Vec<MessageVariant<'src>>,
    /// All original method definitions, preserved for code generation.
    pub methods: Vec<FunctionDef<'src>>,
}

/// A diagnostic emitted by the actor analysis pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ActorDiagnostic<'src> {
    /// Source slice of the problematic construct.
    pub src: &'src str,
    /// Human-readable description of the issue.
    pub message: &'static str,
}

/// Result returned by [`analyse`].
pub struct ActorAnalysisResult<'src> {
    /// One analysis per `actor` block in the module.
    pub analyses: Vec<ActorAnalysis<'src>>,
    /// Diagnostics from the pass.
    pub diagnostics: Vec<ActorDiagnostic<'src>>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Analyse every `actor` definition in `module` and return message-dispatch
/// descriptions for code generation.
///
/// Non-actor items are ignored.
pub fn analyse<'src>(module: &Module<'src>) -> ActorAnalysisResult<'src> {
    let mut analyses    = Vec::new();
    let mut diagnostics = Vec::new();

    for item in &module.items {
        if let ItemKind::ActorDef(ad) = &item.kind {
            analyses.push(build_analysis(ad, &mut diagnostics));
        }
    }

    ActorAnalysisResult { analyses, diagnostics }
}

// ---------------------------------------------------------------------------
// Analysis builder
// ---------------------------------------------------------------------------

fn build_analysis<'src>(
    actor: &ActorDef<'src>,
    diagnostics: &mut Vec<ActorDiagnostic<'src>>,
) -> ActorAnalysis<'src> {
    let messages = build_messages(&actor.methods, diagnostics);

    ActorAnalysis {
        src:           actor.src,
        actor_name:    actor.name,
        msg_enum_name: actor.name.to_string() + "Msg",
        fields:        actor.fields.clone(),
        messages,
        methods:       actor.methods.clone(),
    }
}

/// Build message variants from the actor's methods.
///
/// A method is *dispatchable* if its first parameter is named `self`.
/// Methods without a `self` parameter are skipped with a diagnostic.
fn build_messages<'src>(
    methods: &[FunctionDef<'src>],
    diagnostics: &mut Vec<ActorDiagnostic<'src>>,
) -> Vec<MessageVariant<'src>> {
    let mut variants = Vec::new();

    for method in methods {
        let has_self = method.params.first().map_or(false, |p| p.name == "self");

        if !has_self {
            diagnostics.push(ActorDiagnostic {
                src:     method.src,
                message: "actor method has no `self` parameter and will not be dispatched via message",
            });
            continue;
        }

        // Collect parameters, skipping `self`.
        let params: Vec<MsgParam<'src>> = method.params.iter()
            .skip(1)
            .map(|p| MsgParam {
                name:       p.name,
                annotation: p.annotation.clone(),
            })
            .collect();

        variants.push(MessageVariant {
            method_name:  method.name,
            variant_name: pascal_case(method.name),
            params,
            reply_type: method.return_type.clone(),
            is_async:   method.is_async,
        });
    }

    variants
}

/// Convert `snake_case` (or any casing) to `PascalCase`.
fn pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None    => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn analyse_src(src: &str) -> ActorAnalysisResult<'_> {
        let pr = parser::parse(src);
        analyse(&pr.module)
    }

    #[test]
    fn no_actors_returns_empty() {
        let r = analyse_src("def foo():\n    pass\n");
        assert!(r.analyses.is_empty());
        assert!(r.diagnostics.is_empty());
    }

    #[test]
    fn simple_actor_names() {
        let src = "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n";
        let r = analyse_src(src);
        assert_eq!(r.diagnostics.len(), 0);
        assert_eq!(r.analyses.len(), 1);
        let a = &r.analyses[0];
        assert_eq!(a.actor_name, "Counter");
        assert_eq!(a.msg_enum_name, "CounterMsg");
    }

    #[test]
    fn actor_fields_collected() {
        let src = "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n";
        let r = analyse_src(src);
        let a = &r.analyses[0];
        assert_eq!(a.fields.len(), 1);
        assert_eq!(a.fields[0].name, "count");
    }

    #[test]
    fn actor_message_variants() {
        let src = "actor Counter:\n    var count: u64 = 0\n\n    def increment(self):\n        self.count += 1\n\n    def get(self) -> u64:\n        return self.count\n";
        let r = analyse_src(src);
        assert_eq!(r.diagnostics.len(), 0);
        let a = &r.analyses[0];
        assert_eq!(a.messages.len(), 2);
        assert_eq!(a.messages[0].method_name, "increment");
        assert_eq!(a.messages[0].variant_name, "Increment");
        assert!(a.messages[0].reply_type.is_none());
        assert_eq!(a.messages[1].method_name, "get");
        assert_eq!(a.messages[1].variant_name, "Get");
        assert!(a.messages[1].reply_type.is_some());
    }

    #[test]
    fn method_without_self_emits_diagnostic() {
        let src = "actor Foo:\n    def helper(x: u64):\n        pass\n";
        let r = analyse_src(src);
        assert_eq!(r.diagnostics.len(), 1);
        assert!(r.diagnostics[0].message.contains("no `self`"));
        assert_eq!(r.analyses[0].messages.len(), 0);
    }

    #[test]
    fn method_params_exclude_self() {
        let src = "actor Adder:\n    def add(self, a: u64, b: u64) -> u64:\n        return a + b\n";
        let r = analyse_src(src);
        let a = &r.analyses[0];
        assert_eq!(a.messages[0].params.len(), 2);
        assert_eq!(a.messages[0].params[0].name, "a");
        assert_eq!(a.messages[0].params[1].name, "b");
    }

    #[test]
    fn async_method_is_dispatched() {
        let src = "actor Fetcher:\n    async def fetch(self, url: String) -> String:\n        result = await get(url)\n        return result\n";
        let r = analyse_src(src);
        assert_eq!(r.diagnostics.len(), 0);
        let a = &r.analyses[0];
        assert_eq!(a.messages.len(), 1);
        assert_eq!(a.messages[0].method_name, "fetch");
        assert!(a.messages[0].is_async);
        assert!(a.messages[0].reply_type.is_some());
    }

    #[test]
    fn multiple_actors() {
        let src = "actor A:\n    def ping(self):\n        pass\n\nactor B:\n    def pong(self):\n        pass\n";
        let r = analyse_src(src);
        assert_eq!(r.analyses.len(), 2);
        assert_eq!(r.analyses[0].actor_name, "A");
        assert_eq!(r.analyses[1].actor_name, "B");
    }

    #[test]
    fn msg_enum_name_format() {
        let src = "actor MyService:\n    def run(self):\n        pass\n";
        let r = analyse_src(src);
        assert_eq!(r.analyses[0].msg_enum_name, "MyServiceMsg");
    }

    #[test]
    fn pascal_case_conversion() {
        assert_eq!(pascal_case("increment"), "Increment");
        assert_eq!(pascal_case("get_count"), "GetCount");
        assert_eq!(pascal_case("foo_bar_baz"), "FooBarBaz");
        assert_eq!(pascal_case("already"), "Already");
    }

    #[test]
    fn methods_preserved_on_analysis() {
        let src = "actor Counter:\n    var n: u64 = 0\n\n    def inc(self):\n        self.n += 1\n";
        let r = analyse_src(src);
        assert_eq!(r.analyses[0].methods.len(), 1);
        assert_eq!(r.analyses[0].methods[0].name, "inc");
    }

    #[test]
    fn actor_with_no_fields() {
        let src = "actor Stateless:\n    def ping(self):\n        pass\n";
        let r = analyse_src(src);
        assert_eq!(r.analyses[0].fields.len(), 0);
        assert_eq!(r.analyses[0].messages.len(), 1);
    }

    #[test]
    fn void_method_has_no_reply_type() {
        let src = "actor Logger:\n    def log(self, msg: String):\n        pass\n";
        let r = analyse_src(src);
        let a = &r.analyses[0];
        assert_eq!(a.messages.len(), 1);
        assert!(a.messages[0].reply_type.is_none());
    }

    #[test]
    fn non_void_method_has_reply_type() {
        let src = "actor Store:\n    def read(self, key: String) -> u64:\n        return 0\n";
        let r = analyse_src(src);
        let a = &r.analyses[0];
        assert!(a.messages[0].reply_type.is_some());
    }
}
