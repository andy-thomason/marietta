# channels.ma — producer / consumer pattern using Marietta channels.
#
# `channel<T>` is a first-class type backed by a lock-free MPSC queue.
# The `send` / `recv` operations integrate with the async scheduler so
# producers and consumers run as lightweight coroutines without blocking
# OS threads.
#
# This example shows the planned syntax; channel primitives are implemented
# in step 10 (Runtime) of the compiler roadmap.
#
# Build:  marietta build channels.ma
# Run:    marietta run   channels.ma

# --- Utility functions (fully compiled today) ---

def square(x: u64) -> u64:
    return x * x

def sum_squares(n: u64) -> u64:
    var total: u64 = 0
    var i: u64 = 1
    while i <= n:
        total += square(i)
        i += 1
    return total

# --- Async producer / consumer (planned syntax) ---
#
# async def producer(tx: channel<u64>, n: u64):
#     var i: u64 = 0
#     while i < n:
#         send tx, square(i)
#         i += 1
#
# async def consumer(rx: channel<u64>, n: u64) -> u64:
#     var total: u64 = 0
#     var i: u64 = 0
#     while i < n:
#         let v: u64 = await recv rx
#         total += v
#         i += 1
#     return total
#
# def main():
#     let (tx, rx) = channel<u64>()
#     spawn producer(tx, 10)
#     let result = await consumer(rx, 10)   # result == sum of squares 0..9

def main():
    # Smoke-test the utility functions while channel support is in development.
    var result: u64 = sum_squares(10)
    # sum_squares(10) = 1+4+9+16+25+36+49+64+81+100 = 385
    pass
