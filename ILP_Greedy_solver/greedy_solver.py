"""
Greedy scheduler for snapshot-group placement under the reuse model.

Given per-group execution times and a pairwise reuse matrix, assign groups to
GPU slots across iterations so that the total execution time is minimized.
At each iteration, the algorithm:

  1. Enumerates candidate bottleneck durations by pairing the largest remaining
     group with every other group (reuse-adjusted).
  2. For each candidate, greedily fills the remaining GPU slots using a
     two-pointer search to find the reuse-adjusted pair closest to the target.
  3. Picks the candidate that minimizes iteration waste
         W = D * max(loads) - sum(loads).
  4. Commits the chosen pairs and repeats on the remaining groups.

Variable names follow the paper:
    D           number of GPUs per iteration       -> gpu_num
    R           pairwise reuse matrix              -> reuse
    SG          snapshot-group execution times     -> tasks
    T           per-iteration bottleneck target
    W           per-iteration waste
"""


def _build_hash_table(org_tasks):
    """
    Build a {value -> [indices]} dict mapping each task value back to its
    position(s) in the original (unsorted) task list. Duplicate values are
    queued so that each original index can be claimed independently.
    """
    table = {}
    for idx, t in enumerate(org_tasks):
        table.setdefault(t, []).append(idx)
    return table


def _find_indices(keys, hash_table, indicator, consume=True):
    """
    Look up the original indices (in org_tasks) for the given task values,
    skipping any that are already marked consumed in `indicator`. Zero-valued
    keys are ignored (they correspond to the dummy zero-group).

    Args:
        keys:       iterable of task values to resolve.
        hash_table: dict from _build_hash_table.
        indicator:  list of 0/1 marking already-consumed original indices.
        consume:    if True, marks the matched indices as consumed.

    Returns:
        Sorted list of matched original indices.
    """
    ret = []
    for key in keys:
        if key == 0:
            continue
        for idx in hash_table.get(key, []):
            if indicator[idx] == 0:
                ret.append(idx)
                if consume:
                    indicator[idx] = 1
                break
    ret.sort()
    return ret


def _find_closest_pair_reuse(lst, target, reuse, hash_table, indicator):
    """
    Two-pointer search on a sorted `lst` to find the pair whose reuse-adjusted
    sum is closest to `target`.

    Reuse-adjusted sum for a pair of values (a, b) with original indices
    (i, j) (i < j):
        T' = a + b - R[i][j]

    Args:
        lst:        sorted list of candidate task values.
        target:     T, the target bottleneck duration for this iteration.
        reuse:      R, (n x n) reuse matrix.
        hash_table: dict from _build_hash_table.
        indicator:  list of 0/1 marking consumed original indices.

    Returns:
        (closest_pair, closest_sum):
            closest_pair -- (value_left, value_right) picked from lst
            closest_sum  -- reuse-adjusted sum for that pair
    """
    left, right = 0, len(lst) - 1
    closest_pair = None
    closest_sum = None

    while left < right:
        pair_idx = _find_indices([lst[left], lst[right]],
                                 hash_table, indicator, consume=False)
        if len(pair_idx) == 2:
            current_sum = (lst[left] + lst[right]
                           - reuse[pair_idx[0]][pair_idx[1]])
        else:
            current_sum = lst[left] + lst[right]

        # Always record the reuse-adjusted sum of the chosen pair (the previous
        # version seeded closest_sum with the UN-adjusted lst[left]+lst[right],
        # so a returned pair could drop its reuse benefit from the load).
        if (closest_pair is None
                or abs(current_sum - target) < abs(closest_sum - target)
                or (abs(current_sum - target) == abs(closest_sum - target)
                    and current_sum > closest_sum)):
            closest_sum = current_sum
            closest_pair = (lst[left], lst[right])

        if current_sum < target:
            left += 1        # under target: advance the left pointer
        else:
            right -= 1       # over/equal: retreat the right pointer

    return closest_pair, closest_sum


def _cal_diff(lst):
    """Waste of one iteration's per-GPU loads: D * max(lst) - sum(lst)."""
    return max(lst) * len(lst) - sum(lst)


def greedy(tasks, gpu_num, lat, reuse):
    """
    Greedy scheduler that selects reuse-maximizing group pairs per iteration.

    Args:
        tasks:    list of snapshot-group execution times t_k.
        gpu_num:  D, number of GPUs per iteration.
        lat:      per-iteration overhead alpha (added to iteration time).
        reuse:    (n x n) reuse matrix R.

    Returns:
        total_time:  sum_i (max_j load_{i,j} + alpha)
        waste:       sum of per-iteration D*max - sum
        cb_index:    schedule layout (list of original-index groups per slot)
        imbalance:   total_time / min_time
    """
    org_tasks = list(tasks)
    hash_table = _build_hash_table(org_tasks)
    indicator = [0 for _ in range(len(org_tasks))]

    # Insert a zero sentinel so the "run largest alone" case appears naturally
    # as one of the candidate targets, then sort ascending.
    tasks = sorted([0.0] + list(tasks))

    cb_tasks = []
    cb_index = []

    # Loop while a full D-slot iteration can still be formed from REAL groups.
    # The `any(t != 0 ...)` guard stops once only the persistent zero sentinel
    # remains (needed because the sentinel is never removed, e.g. for D == 1).
    while len(tasks) >= gpu_num and any(t != 0 for t in tasks):
        n = len(tasks)
        largest = tasks[-1]

        best_W = float('inf')
        best_schedule = None   # (loads, pair_vals, smaller, largest)

        # Enumerate candidate targets: pair the largest group with each other
        # group, reuse-adjusted. Pick the candidate with minimum waste.
        for i in range(n - 1):
            smaller = tasks[i]

            # Probe reuse(smaller, largest) without consuming the indicator.
            pair_idx = _find_indices([smaller, largest], hash_table,
                                     indicator, consume=False)
            if len(pair_idx) == 2:
                a, b = sorted(pair_idx)
                R_il = reuse[a][b]
            else:
                R_il = 0.0

            target = smaller + largest - R_il

            # Fill the remaining D-1 GPU slots by searching for pairs whose
            # reuse-adjusted sum is closest to `target`. The zero sentinel
            # is already in `tasks`; when i == 0 it has been excluded from
            # `rest`, forcing the other slots to use non-zero pairs.
            rest = [t for idx, t in enumerate(tasks)
                    if idx != i and idx != n - 1]
            lst = list(rest)
            loads = [target]
            pair_vals = []

            for _ in range(gpu_num - 1):
                if len(lst) < 2:
                    # not enough items left for another pair; pad with zeros
                    loads.append(0.0)
                    pair_vals.append([0, 0])
                    continue
                pair, clo_sum = _find_closest_pair_reuse(
                    lst, target, reuse, hash_table, indicator)
                loads.append(clo_sum)
                pair_vals.append([pair[0], pair[1]])
                if pair[0] != 0:
                    lst.remove(pair[0])
                if pair[1] != 0:
                    lst.remove(pair[1])

            # Iteration waste: D * max(loads) - sum(loads)
            W = gpu_num * max(loads) - sum(loads)

            if W < best_W:
                best_W = W
                best_schedule = (loads, pair_vals, smaller, largest)

        # Commit the winning candidate's pairs.
        loads, pair_vals, smaller, largest = best_schedule
        cb_tasks.append(loads)

        # Record indices in schedule order (target pair first, then others).
        cb_index.append(_find_indices([smaller, largest], hash_table,
                                      indicator, consume=True))
        for pv in pair_vals:
            cb_index.append(_find_indices([pv[0], pv[1]], hash_table,
                                          indicator, consume=True))

        # Remove the consumed values from `tasks` (by value). Keep the zero
        # sentinel persistent so every iteration retains the "run a group alone"
        # option (paper's Preprocess semantics); only remove it if `smaller` is
        # a real group.
        if smaller != 0:
            tasks.remove(smaller)
        tasks.remove(largest)
        for pv in pair_vals:
            for val in pv:
                if val != 0 and val in tasks:
                    tasks.remove(val)

    # Tail iteration: fewer than D real groups remain; place them one-per-GPU.
    # Exclude the persistent zero sentinel so no phantom all-zero iteration is
    # appended when only the sentinel is left.
    tail = [t for t in tasks if t != 0]
    if len(tail) != 0:
        padded = tail + [0] * (gpu_num - len(tail))
        cb_tasks.append(padded)
        for t in tail:
            cb_index.append(_find_indices([t], hash_table,
                                          indicator, consume=True))

    # Summary metrics.
    total_time = 0.0
    min_time = 0.0
    waste = 0.0
    for ts in cb_tasks:
        total_time += max(ts) + lat
        min_time += min(ts) + lat
        waste += _cal_diff(ts)

    imbalance = [0.0 for _ in range(gpu_num)]
    for ts in cb_tasks:
        for i, t in enumerate(ts):
            imbalance[i] += t

    # Drop empty slots (from dummy zero-group padding). A zero-length slot would
    # be misread by the training driver as a paired slot and crash on indexing,
    # so never expose one to consumers.
    cb_index = [slot for slot in cb_index if slot]

    # imbalance = makespan / best-case; guard against min_time == 0 (possible
    # when lat == 0 and an iteration is zero-padded).
    imbalance_ratio = total_time / min_time if min_time > 0 else float('inf')
    return total_time, waste, cb_index, imbalance_ratio


if __name__ == "__main__":
    import time

    # Example input taken from greedy_solver.py (arxiv / evolvegcn, D=4).
    gpu_num = 4
    alpha = 0.005
    # get_latency(gpu_num, alpha) + 1.5 from greedy_solver.py
    latency = 2 * (gpu_num - 2) * alpha + 1.5

    # Per-snapshot-group execution times t_k (length n).
    # Load from your profiling log, e.g. via read_from_file(...).
    tasks = YOUR_SNAPSHOT_GROUP_TIMES  # e.g. [t_1, t_2, ..., t_n]

    # Upper-triangular n x n reuse matrix R; R[i][j] is the reuse benefit
    # (shared computation / transfer savings) when groups i and j co-locate
    # on the same GPU in the same iteration. Only i < j entries are used.
    reuse = YOUR_REUSE_MATRIX  # e.g. [[0, R_01, R_02, ...], [0, 0, R_12, ...], ...]


    s_t = time.time()
    total, waste, indices, ib = greedy(
        tasks, gpu_num, latency, reuse)
    print(f'elapsed: {time.time() - s_t:.3f}s')
    print('indices:', indices)
