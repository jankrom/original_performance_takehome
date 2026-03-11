"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _scratch_range(self, start, length):
        return set(range(start, start + length))

    def _slot_meta(self, engine, slot):
        reads = set()
        writes = set()
        mem_read = False
        mem_write = False
        barrier = False

        match (engine, slot):
            case ("alu", (_, dest, a1, a2)):
                writes.add(dest)
                reads.update((a1, a2))
            case ("valu", ("vbroadcast", dest, src)):
                writes.update(self._scratch_range(dest, VLEN))
                reads.add(src)
            case ("valu", ("multiply_add", dest, a, b, c)):
                writes.update(self._scratch_range(dest, VLEN))
                reads.update(self._scratch_range(a, VLEN))
                reads.update(self._scratch_range(b, VLEN))
                reads.update(self._scratch_range(c, VLEN))
            case ("valu", (_, dest, a1, a2)):
                writes.update(self._scratch_range(dest, VLEN))
                reads.update(self._scratch_range(a1, VLEN))
                reads.update(self._scratch_range(a2, VLEN))
            case ("load", ("load", dest, addr)):
                writes.add(dest)
                reads.add(addr)
                mem_read = True
            case ("load", ("load_offset", dest, addr, offset)):
                writes.add(dest + offset)
                reads.add(addr + offset)
                mem_read = True
            case ("load", ("vload", dest, addr)):
                writes.update(self._scratch_range(dest, VLEN))
                reads.add(addr)
                mem_read = True
            case ("load", ("const", dest, _)):
                writes.add(dest)
            case ("store", ("store", addr, src)):
                reads.update((addr, src))
                mem_write = True
            case ("store", ("vstore", addr, src)):
                reads.add(addr)
                reads.update(self._scratch_range(src, VLEN))
                mem_write = True
            case ("flow", ("select", dest, cond, a, b)):
                writes.add(dest)
                reads.update((cond, a, b))
            case ("flow", ("add_imm", dest, a, _)):
                writes.add(dest)
                reads.add(a)
            case ("flow", ("vselect", dest, cond, a, b)):
                writes.update(self._scratch_range(dest, VLEN))
                reads.update(self._scratch_range(cond, VLEN))
                reads.update(self._scratch_range(a, VLEN))
                reads.update(self._scratch_range(b, VLEN))
            case ("flow", ("coreid", dest)):
                writes.add(dest)
            case ("flow", ("trace_write", val)):
                reads.add(val)
            case ("flow", ("cond_jump", cond, _)):
                reads.add(cond)
                barrier = True
            case ("flow", ("cond_jump_rel", cond, _)):
                reads.add(cond)
                barrier = True
            case ("flow", ("jump_indirect", addr)):
                reads.add(addr)
                barrier = True
            case ("flow", ("pause",) | ("halt",) | ("jump", _)):
                barrier = True
            case ("debug", ("compare", loc, _)):
                reads.add(loc)
            case ("debug", ("vcompare", loc, _)):
                reads.update(self._scratch_range(loc, VLEN))
            case ("debug", _):
                pass
            case _:
                raise NotImplementedError(f"Unknown slot for scheduling: {engine} {slot}")

        return {
            "reads": reads,
            "writes": writes,
            "mem_read": mem_read,
            "mem_write": mem_write,
            "barrier": barrier,
        }

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        bundle = {}
        bundle_reads = set()
        bundle_writes = set()
        bundle_has_mem_write = False

        def flush():
            nonlocal bundle, bundle_reads, bundle_writes, bundle_has_mem_write
            if bundle:
                instrs.append(bundle)
                bundle = {}
                bundle_reads = set()
                bundle_writes = set()
                bundle_has_mem_write = False

        for engine, slot in slots:
            meta = self._slot_meta(engine, slot)
            engine_slots = bundle.get(engine, [])
            exceeds_slots = len(engine_slots) >= SLOT_LIMITS[engine]
            depends_on_pending_write = bool(meta["reads"] & bundle_writes)
            overwrites_pending_write = bool(meta["writes"] & bundle_writes)
            conflicts_with_store = bundle_has_mem_write and (meta["mem_read"] or meta["mem_write"])
            is_barrier = meta["barrier"]

            if (
                exceeds_slots
                or depends_on_pending_write
                or overwrites_pending_write
                or conflicts_with_store
                or is_barrier
            ):
                flush()

            if is_barrier:
                instrs.append({engine: [slot]})
                continue

            bundle.setdefault(engine, []).append(slot)
            bundle_reads.update(meta["reads"])
            bundle_writes.update(meta["writes"])
            bundle_has_mem_write = bundle_has_mem_write or meta["mem_write"]

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
