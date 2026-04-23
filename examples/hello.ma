# hello.ma — the classic first program, Marietta style.
#
# Marietta compiles to native code via Cranelift.  The `run` subcommand
# JIT-compiles the file and calls `main()` if one is defined.
#
# Build:   marietta build hello.ma
# Run:     marietta run   hello.ma

def greet(name: u64) -> u64:
    # In a future release `name` will be a String.  For now we accept a code
    # point and echo it back so the example is fully executable today.
    return name

def add(a: u64, b: u64) -> u64:
    return a + b

def main():
    var result: u64 = add(40, 2)
    pass
