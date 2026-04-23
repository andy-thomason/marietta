# counter_actor.ma — a simple counter actor demonstrating the actor model.
#
# Actors in Marietta compile to a struct + message enum.  Each method call
# becomes a channel send; the runtime dispatches messages on a per-actor task.
#
# This example shows the intended syntax; the runtime and full actor lowering
# are implemented in steps 7 and 10 of the compiler roadmap.
#
# Build:  marietta build counter_actor.ma
# Run:    marietta run   counter_actor.ma

actor Counter:
    var count: u64 = 0

    def increment(self):
        self.count += 1

    def add(self, n: u64):
        self.count += n

    def reset(self):
        self.count = 0

    def get(self) -> u64:
        return self.count

# A standalone helper to demonstrate that actors and free functions coexist.
def clamp(value: u64, lo: u64, hi: u64) -> u64:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value

def main():
    # In the full runtime, actors are spawned and methods are sent as messages:
    #   let c = spawn Counter
    #   send c.increment()
    #   send c.add(5)
    #   let n = await c.get()   # n == 6
    #
    # For now we exercise the compiled actor methods directly via function calls.
    var v: u64 = clamp(15, 0, 10)
    pass
