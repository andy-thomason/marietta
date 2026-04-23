# dyn_shape.ma — dynamic dispatch through a trait object (dyn Trait).
#
# This example demonstrates Marietta's trait system.  Each `impl Trait for Type`
# block teaches the compiler to:
#   1. Mangle the method names: `Circle__area`, `Square__area` etc.
#   2. Emit a vtable entry: `Circle__Shape__vtable = [&Circle__area, &Circle__perimeter]`
#
# A `dyn Shape` parameter is a fat pointer — a 16-byte stack block:
#   offset 0: data_ptr   (pointer to the concrete struct data)
#   offset 8: vtable_ptr (pointer to the matching vtable)
#
# Calling `s.area()` on a `dyn Shape` compiles to:
#   data_ptr   = *(fat_ptr + 0)
#   vtable_ptr = *(fat_ptr + 8)
#   fn_ptr     = *(vtable_ptr + 0)      # method 0 = area
#   result     = fn_ptr(data_ptr)
#
# Build:  marietta build dyn_shape.ma
# Run:    marietta run   dyn_shape.ma

# ---------------------------------------------------------------------------
# 1.  Define a trait
# ---------------------------------------------------------------------------

trait Shape:
    def area(self) -> u64:
    def perimeter(self) -> u64:

# ---------------------------------------------------------------------------
# 2.  Define two concrete types
# ---------------------------------------------------------------------------

struct Circle:
    radius: u64

struct Square:
    side: u64

# ---------------------------------------------------------------------------
# 3.  Implement the trait for each type
#
#     These compile to mangled functions:
#       Circle__area, Circle__perimeter
#       Square__area, Square__perimeter
#     and register two vtables in the module:
#       Circle__Shape__vtable  = [&Circle__area, &Circle__perimeter]
#       Square__Shape__vtable  = [&Square__area, &Square__perimeter]
# ---------------------------------------------------------------------------

impl Shape for Circle:
    def area(self) -> u64:
        return self.radius * self.radius

    def perimeter(self) -> u64:
        return self.radius * 4

impl Shape for Square:
    def area(self) -> u64:
        return self.side * self.side

    def perimeter(self) -> u64:
        return self.side * 4

# ---------------------------------------------------------------------------
# 4.  A function that accepts any Shape via dynamic dispatch
#
#     The parameter `s: dyn Shape` arrives as an I64 address of the fat
#     pointer block.  Each method call compiles to a VtableCall instruction
#     (indirect dispatch through the vtable pointer).
# ---------------------------------------------------------------------------

def total(s: dyn Shape) -> u64:
    return s.area() + s.perimeter()

# ---------------------------------------------------------------------------
# 5.  main
#
#     Direct concrete calls work today (static dispatch via mangled names).
#     Passing a concrete struct *as* a dyn Shape requires fat-pointer
#     construction, which will be wired up when struct-allocation syntax is
#     added.  The `total` function above already compiles with the correct
#     VtableCall IR; only the caller-side coercion is pending.
# ---------------------------------------------------------------------------

def main():
    # These call the mangled concrete implementations directly.
    var c: Circle
    c.radius = 3
    var c_area: u64 = Circle__area(c)        # static dispatch: r*r = 9
    var c_peri: u64 = Circle__perimeter(c)   # static dispatch: r*4 = 12

    var sq: Square
    sq.side = 4
    var sq_area: u64 = Square__area(sq)      # static dispatch: s*s = 16
    var sq_peri: u64 = Square__perimeter(sq) # static dispatch: s*4 = 16

    # Future: once struct-allocation and fat-pointer coercion are in place,
    # these will compile and dispatch through the vtable:
    #   var result_c:  u64 = total(c)
    #   var result_sq: u64 = total(sq)

    pass
