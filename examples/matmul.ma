# matmul.ma — matrix multiply over f64.
#
# Matrices are represented as flat u64 "pointers" (indices into a conceptual
# heap allocation).  Full slice / tensor support is planned for a later
# compiler release.
#
# Build:  marietta build matmul.ma
# Run:    marietta run   matmul.ma

# Multiply two 2×2 matrices stored as (a00, a01, a10, a11) … using scalars
# until the compiler gains first-class array / slice support.

def matmul2x2(
    a00: f64, a01: f64, a10: f64, a11: f64,
    b00: f64, b01: f64, b10: f64, b11: f64,
) -> f64:
    # Returns just the top-left element c00 as a smoke-test.
    # Full result would require a tuple or out-parameter (upcoming feature).
    var c00: f64 = a00 * b00 + a01 * b10
    return c00

def dot(ax: f64, ay: f64, bx: f64, by: f64) -> f64:
    return ax * bx + ay * by

def main():
    # Identity × anything = anything.
    # [1 0] × [5 6] = [5 6]
    # [0 1]   [7 8]   [7 8]
    var c00: f64 = matmul2x2(
        1.0, 0.0, 0.0, 1.0,
        5.0, 6.0, 7.0, 8.0,
    )
    # c00 == 5.0
    pass
