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
        self.init_slots = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}

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
            self.init_slots.append(("load", ("const", addr, val)))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vec_const(self, val, name=None):
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val, name)
            vec_addr = self.alloc_scratch(f"{name}_vec" if name is not None else None, VLEN)
            self.init_slots.append(("valu", ("vbroadcast", vec_addr, scalar_addr)))
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_hash_addr, tmp1, tmp2, vec_hash_consts):
        slots = []

        for stage in vec_hash_consts:
            if stage[0] == "multiply_add":
                _, multiplier_vec, add_vec = stage
                slots.append(
                    ("valu", ("multiply_add", val_hash_addr, val_hash_addr, multiplier_vec, add_vec))
                )
                continue

            _, op1, op2, op3, val1_vec, val3_vec = stage
            slots.append(("valu", (op1, tmp1, val_hash_addr, val1_vec)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, val3_vec)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_block_bundles(
        self,
        idx_addr,
        val_addr,
        gather_addr,
        node_addr,
        tmp1,
        tmp2,
        forest_values_p_v,
        one_v,
        two_v,
        n_nodes_v,
        vec_hash_consts,
        store_addr=None,
        update_index=True,
    ):
        bundles = [
            {"valu": [("+", gather_addr, idx_addr, forest_values_p_v)]},
            {
                "load": [
                    ("load_offset", node_addr, gather_addr, 0),
                    ("load_offset", node_addr, gather_addr, 1),
                ]
            },
            {
                "load": [
                    ("load_offset", node_addr, gather_addr, 2),
                    ("load_offset", node_addr, gather_addr, 3),
                ]
            },
            {
                "load": [
                    ("load_offset", node_addr, gather_addr, 4),
                    ("load_offset", node_addr, gather_addr, 5),
                ]
            },
            {
                "load": [
                    ("load_offset", node_addr, gather_addr, 6),
                    ("load_offset", node_addr, gather_addr, 7),
                ]
            },
            {"valu": [("^", val_addr, val_addr, node_addr)]},
        ]

        for stage in vec_hash_consts:
            if stage[0] == "multiply_add":
                _, multiplier_vec, add_vec = stage
                bundles.append(
                    {
                        "valu": [
                            ("multiply_add", val_addr, val_addr, multiplier_vec, add_vec)
                        ]
                    }
                )
                continue

            _, op1, op2, op3, val1_vec, val3_vec = stage
            bundles.append(
                {
                    "valu": [
                        (op1, tmp1, val_addr, val1_vec),
                        (op3, tmp2, val_addr, val3_vec),
                    ]
                }
            )
            bundles.append({"valu": [(op2, val_addr, tmp1, tmp2)]})

        if update_index:
            bundles.extend(
                [
                    {
                        "valu": [
                            ("&", node_addr, val_addr, one_v),
                            ("multiply_add", idx_addr, idx_addr, two_v, one_v),
                        ]
                    },
                    {"valu": [("+", idx_addr, idx_addr, node_addr)]},
                    {"valu": [("<", node_addr, idx_addr, n_nodes_v)]},
                    {"valu": [("*", idx_addr, idx_addr, node_addr)]},
                ]
            )
        if store_addr is not None:
            bundles.append({"store": [("vstore", store_addr, val_addr)]})
        return bundles

    def _bundle_fits(self, bundle, candidate):
        for engine, slots in candidate.items():
            if len(bundle.get(engine, [])) + len(slots) > SLOT_LIMITS[engine]:
                return False
        return True

    def _merge_bundle(self, bundle, candidate):
        for engine, slots in candidate.items():
            bundle.setdefault(engine, []).extend(slots)

    def schedule_bundle_groups(self, bundle_groups):
        instrs = []
        positions = [0] * len(bundle_groups)
        start = 0

        while any(pos < len(group) for pos, group in zip(positions, bundle_groups)):
            instr = {}

            load_candidates = []
            valu_candidates = []
            other_candidates = []

            for offset in range(len(bundle_groups)):
                group_i = (start + offset) % len(bundle_groups)
                if positions[group_i] >= len(bundle_groups[group_i]):
                    continue

                candidate = bundle_groups[group_i][positions[group_i]]
                if "load" in candidate:
                    load_candidates.append(group_i)
                elif "valu" in candidate:
                    valu_candidates.append(group_i)
                else:
                    other_candidates.append(group_i)

            if load_candidates:
                group_i = min(load_candidates, key=lambda i: positions[i])
                candidate = bundle_groups[group_i][positions[group_i]]
                self._merge_bundle(instr, candidate)
                positions[group_i] += 1

            for group_i in sorted(valu_candidates, key=lambda i: positions[i], reverse=True):
                candidate = bundle_groups[group_i][positions[group_i]]
                if self._bundle_fits(instr, candidate):
                    self._merge_bundle(instr, candidate)
                    positions[group_i] += 1

            for group_i in other_candidates:
                candidate = bundle_groups[group_i][positions[group_i]]
                if self._bundle_fits(instr, candidate):
                    self._merge_bundle(instr, candidate)
                    positions[group_i] += 1

            instrs.append(instr)
            start = (start + 1) % len(bundle_groups)

        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation that keeps batch state in scratch and writes
        the output values back once at the end.
        """
        assert batch_size % VLEN == 0, "This kernel expects a full vector tail"

        one_v = self.scratch_vec_const(1, "one")
        two_v = self.scratch_vec_const(2, "two")
        forest_values_p_v = self.scratch_vec_const(7, "forest_values_p")
        n_nodes_v = self.scratch_vec_const(n_nodes, "n_nodes")

        inp_indices_p = 7 + n_nodes
        inp_values_p = inp_indices_p + batch_size

        vec_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                vec_hash_consts.append(
                    (
                        "multiply_add",
                        self.scratch_vec_const((1 << val3) + 1, f"hash_{hi}_mul"),
                        self.scratch_vec_const(val1, f"hash_{hi}_add"),
                    )
                )
            else:
                vec_hash_consts.append(
                    (
                        "generic",
                        op1,
                        op2,
                        op3,
                        self.scratch_vec_const(val1, f"hash_{hi}_lhs"),
                        self.scratch_vec_const(val3, f"hash_{hi}_rhs"),
                    )
                )

        prologue = []  # array of slots

        idx_base = self.alloc_scratch("idx", batch_size)
        val_base = self.alloc_scratch("val", batch_size)
        wave_size = batch_size // VLEN
        gather_bank = self.alloc_scratch("gather_bank", wave_size * VLEN)
        node_bank = self.alloc_scratch("node_bank", wave_size * VLEN)
        tmp1_bank = self.alloc_scratch("tmp1_bank", wave_size * VLEN)

        for i in range(0, batch_size, VLEN):
            prologue.append(("load", ("vload", val_base + i, self.scratch_const(inp_values_p + i))))

        bundle_groups = []
        for block_i, i in enumerate(range(0, batch_size, VLEN)):
            buf = block_i * VLEN
            block_bundles = []
            for round_i in range(rounds):
                store_addr = None
                if round_i == rounds - 1:
                    store_addr = self.scratch_const(inp_values_p + i)
                block_bundles.extend(
                    self.build_block_bundles(
                        idx_base + i,
                        val_base + i,
                        gather_bank + buf,
                        node_bank + buf,
                        tmp1_bank + buf,
                        gather_bank + buf,
                        forest_values_p_v,
                        one_v,
                        two_v,
                        n_nodes_v,
                        vec_hash_consts,
                        store_addr,
                        round_i != rounds - 1,
                    )
                )
            bundle_groups.append(block_bundles)

        self.instrs.extend(self.build(self.init_slots))
        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        self.instrs.extend(self.build(prologue))
        self.instrs.extend(self.schedule_bundle_groups(bundle_groups))

        self.instrs[-1]["flow"] = [("pause",)]

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
