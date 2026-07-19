"""
build_strategy.py -- bridge from the profiler's reuse log to a training strategy
file (pipeline stage T3 -> T4).

The schedulers (REAL-G greedy / REAL-L ILP) operate on GROUP indices 0..n-1,
but training.py reads each integer in the strategy file as a snapshot START
index. The profiler emits a companion ``<reuse_log>.groups.txt`` that maps group
index -> snapshot start index. This script:

  1. loads ``time_costs`` + ``reuse`` from the profiler's reuse log,
  2. loads the group->start map from ``<reuse_log>.groups.txt``,
  3. runs the chosen solver to get a schedule of GROUP-index slots,
  4. drops empty slots and maps group indices -> snapshot start indices,
  5. writes a strategy file whose per-model line training.py consumes.

Example::

    python build_strategy.py --reuse-log arxiv_wdgcn_reuse.txt \
        --solver greedy --gpus 8 --model wdgcn --out strategy.txt
"""

import argparse
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "ILP_Greedy_solver"))

# training.py's per-model strategy line (1-indexed).
_MODEL_STRATEGY_LINE = {"evolvegcn": 2, "wdgcn": 4, "tgcn": 6, "gatlstm": 8}
_N_LINES = 8


def read_reuse_log(path):
    """Parse ``[time_costs]\\n\\n[reuse]`` (same format the ILP solver reads)."""
    with open(path) as f:
        parts = f.read().strip().split("\n\n")
    time_costs = ast.literal_eval(parts[0])
    reuse = ast.literal_eval(parts[1])
    return time_costs, reuse


def read_groups(path):
    """Parse ``<reuse_log>.groups.txt`` -> list of snapshot start indices."""
    with open(path) as f:
        return [int(x) for x in ast.literal_eval(f.read().strip())]


def solve(solver, time_costs, reuse, gpus, alpha, gap):
    """Return a schedule as a list of slots, each a list of GROUP indices."""
    n = len(time_costs)
    if solver == "greedy":
        from greedy_solver import greedy
        # lat is a per-iteration constant; it does not affect the chosen schedule.
        _, _, cb_index, _ = greedy(time_costs, gpus, alpha, reuse)
        schedule = cb_index
    elif solver == "ilp":
        from ilp_solver_gurobi import get_snapshot_alloc_reuse
        _, assign_list = get_snapshot_alloc_reuse(
            n, gpus, time_costs, reuse, L=2, alpha=alpha, ideal_gap=gap)
        schedule = assign_list
    else:
        raise ValueError(f"unknown solver: {solver}")
    # Drop empty slots defensively (greedy already filters, ILP never emits them).
    return [slot for slot in schedule if slot]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-log", required=True,
                    help="profiler output (time_costs + reuse matrix)")
    ap.add_argument("--groups", default=None,
                    help="group->start map (default: <reuse-log>.groups.txt)")
    ap.add_argument("--solver", choices=["greedy", "ilp"], default="greedy")
    ap.add_argument("--gpus", type=int, required=True, help="D, GPUs per iteration")
    ap.add_argument("--world-size", type=int, default=1,
                    help="training world_size; slot count is balanced to a "
                         "multiple of this so per-rank barriers stay aligned")
    ap.add_argument("--model", choices=list(_MODEL_STRATEGY_LINE), required=True)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="per-iteration sync overhead")
    ap.add_argument("--gap", type=float, default=0.05, help="ILP optimality gap")
    ap.add_argument("--out", default="strategy.txt")
    args = ap.parse_args()

    groups_path = args.groups or (args.reuse_log + ".groups.txt")
    time_costs, reuse = read_reuse_log(args.reuse_log)
    starts = read_groups(groups_path)
    if len(starts) != len(time_costs):
        raise ValueError(f"groups.txt has {len(starts)} starts but reuse log has "
                         f"{len(time_costs)} groups")

    schedule_groups = solve(args.solver, time_costs, reuse, args.gpus,
                            args.alpha, args.gap)

    # Balance the slot count to a multiple of world_size so each rank receives
    # the same number of slots. Paired slots are split into singletons (which
    # preserves coverage) until the count aligns.
    if args.world_size > 1:
        def _aligned(s):
            return len(s) % args.world_size == 0 and len(s) >= args.world_size
        while not _aligned(schedule_groups):
            paired = next((i for i, s in enumerate(schedule_groups) if len(s) == 2), None)
            if paired is None:
                print(f"[build_strategy] WARN: cannot balance {len(schedule_groups)} "
                      f"slots to world_size={args.world_size} (no paired slots to split)")
                break
            a, b = schedule_groups.pop(paired)
            schedule_groups += [[a], [b]]

    # Map GROUP indices -> snapshot START indices (the P4 translation).
    schedule_starts = [[starts[g] for g in slot] for slot in schedule_groups]

    # sanity: every group scheduled exactly once
    flat = [g for slot in schedule_groups for g in slot]
    assert sorted(flat) == list(range(len(time_costs))), \
        f"schedule does not cover every group exactly once: {sorted(flat)}"

    # Write the strategy file: the schedule on this model's line, safe singleton
    # fillers on the others (training.py reads only the model's line).
    target = _MODEL_STRATEGY_LINE[args.model]
    filler = f"[[{starts[0]}]]"
    lines = []
    for ln in range(1, _N_LINES + 1):
        if ln == target:
            lines.append(f"{schedule_starts!r}  # {args.model} "
                         f"({args.solver}, {args.gpus} GPUs, group->start mapped)")
        else:
            lines.append(filler)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_paired = sum(1 for s in schedule_starts if len(s) == 2)
    print(f"[build_strategy] solver={args.solver} model={args.model} gpus={args.gpus}")
    print(f"[build_strategy] {len(time_costs)} groups -> {len(schedule_starts)} slots "
          f"({n_paired} paired, {len(schedule_starts)-n_paired} singleton)")
    print(f"[build_strategy] group->start map: {starts}")
    print(f"[build_strategy] schedule (snapshot starts): {schedule_starts}")
    print(f"[build_strategy] wrote {args.out} (line {target} for {args.model})")


if __name__ == "__main__":
    main()
