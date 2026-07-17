import sys
import random

from gurobipy import *


gap = 0.02


def read_from_file(filename):
    import ast
    with open(filename, 'r') as f:
        content = f.read()
    parts = content.strip().split('\n\n')
    time_costs = ast.literal_eval(parts[0])
    reuse = ast.literal_eval(parts[1])
    return time_costs, reuse


def mycallback(model, where):
    if where == GRB.Callback.MIP:
        best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        best_solution = model.cbGet(GRB.Callback.MIP_OBJBST)
        if best_solution - best_bound <= abs(best_solution * gap):
            print("Terminating early due to small improvement.")
            model.terminate()


# =============================================================================
# ILP with reuse -- epigraph min-max linearization (Method-v5.tex Sec. 5.2).
# =============================================================================
def get_snapshot_alloc_reuse(snapshot_n, gpu_n, time_costs, reuse,
                             L=2, alpha=0.0, ideal_gap=0.02):
    """
    Solve the ILP with data reuse (paper Eq. 'obj'..'z-cont').

    Args:
        snapshot_n: n, number of snapshot groups.
        gpu_n:      G, number of GPUs.
        time_costs: list of t_k, length n.
        reuse:      (n x n) matrix of R_{k1,k2} (only k1<k2 entries used).
        L:          GPU capacity per iteration (paper default 2).
        alpha:      per-iteration overhead (e.g. allreduce latency).
        ideal_gap:  relative MIP gap to early-stop at.
    """
    global gap
    gap = ideal_gap

    model = Model("minimize_training_time_with_reuse")
    model.setParam("Threads", 12)
    model.setParam("Presolve", 2)
    model.setParam("Heuristics", 0.3)
    model.setParam("Cuts", 0)
    model.setParam("Symmetry", 0)

    max_iter_n = (snapshot_n + gpu_n - 1) // gpu_n   # m = ceil(n / G)
    sum_t = sum(time_costs)                           # upper bound on per-GPU load / makespan (variable bounds)

    # -------------------------------------------------------------------------
    # Decision variables
    # -------------------------------------------------------------------------
    # x_{k,i,j}: binary (paper Eq. x)
    x = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
           for j in range(gpu_n)]
          for i in range(max_iter_n)]
         for k in range(snapshot_n)]

    # y_{k1,k2,i,j}: binary, only for k1 < k2 (paper Eq. y)
    y = {}
    for k1 in range(snapshot_n):
        for k2 in range(k1 + 1, snapshot_n):
            for i in range(max_iter_n):
                for j in range(gpu_n):
                    y[k1, k2, i, j] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{k1}_{k2}_{i}_{j}")

    # Per-GPU effective time (load) t_gpu_{i,j} = base cost - reuse benefit
    # (paper Eq. L_i_j).
    t_gpu = [[model.addVar(lb=-sum_t, ub=sum_t, vtype=GRB.CONTINUOUS,
                           name=f"t_gpu_{i}_{j}")
              for j in range(gpu_n)]
             for i in range(max_iter_n)]

    # Per-iteration makespan T_i (epigraph variable, paper Eq. minmax_constraint).
    # Minimizing sum_i T_i drives each T_i to max_j t_gpu_{i,j} + alpha with no
    # Big-M and no auxiliary binaries.
    T = [model.addVar(lb=0, ub=sum_t + alpha, vtype=GRB.CONTINUOUS, name=f"T_{i}")
         for i in range(max_iter_n)]

    # -------------------------------------------------------------------------
    # Constraint: assignment (paper Eq. assign)
    #   sum_{i,j} x_{k,i,j} = 1, forall k
    # -------------------------------------------------------------------------
    for k in range(snapshot_n):
        model.addConstr(
            quicksum(x[k][i][j] for i in range(max_iter_n)
                                for j in range(gpu_n)) == 1,
            name=f"assign_once_{k}")

    # -------------------------------------------------------------------------
    # Constraint: GPU capacity L (paper Eq. cap)
    #   sum_k x_{k,i,j} <= L, forall i, j
    # -------------------------------------------------------------------------
    for i in range(max_iter_n):
        for j in range(gpu_n):
            model.addConstr(
                quicksum(x[k][i][j] for k in range(snapshot_n)) <= L,
                name=f"gpu_capacity_{i}_{j}")

    # -------------------------------------------------------------------------
    # Constraint: y linearization (paper Eq. y1, y2, y3)
    #   y <= x_{k1}, y <= x_{k2}, y >= x_{k1} + x_{k2} - 1
    # -------------------------------------------------------------------------
    for k1 in range(snapshot_n):
        for k2 in range(k1 + 1, snapshot_n):
            for i in range(max_iter_n):
                for j in range(gpu_n):
                    model.addConstr(y[k1, k2, i, j] <= x[k1][i][j])
                    model.addConstr(y[k1, k2, i, j] <= x[k2][i][j])
                    model.addConstr(
                        y[k1, k2, i, j] >= x[k1][i][j] + x[k2][i][j] - 1)

    # -------------------------------------------------------------------------
    # Define per-GPU time (paper Eq. z-def)
    #   t_gpu_{i,j} = sum_k x_{k,i,j} t_k
    #                 - sum_{k1<k2} y_{k1,k2,i,j} R_{k1,k2}
    # -------------------------------------------------------------------------
    for i in range(max_iter_n):
        for j in range(gpu_n):
            model.addConstr(
                t_gpu[i][j] ==
                quicksum(x[k][i][j] * time_costs[k] for k in range(snapshot_n))
                - quicksum(
                    y[k1, k2, i, j] * reuse[k1][k2]
                    for k1 in range(snapshot_n)
                    for k2 in range(k1 + 1, snapshot_n)),
                name=f"def_t_gpu_{i}_{j}")

    # -------------------------------------------------------------------------
    # Min-max via epigraph (paper Eq. minmax_constraint):
    #   T_i >= t_gpu_{i,j} + alpha,   forall j
    # Because the objective minimizes sum_i T_i, each T_i is tightened to
    # max_j t_gpu_{i,j} + alpha automatically -- no Big-M, no auxiliary binaries.
    # This is the tightened LP relaxation the paper argues for.
    # -------------------------------------------------------------------------
    for i in range(max_iter_n):
        for j in range(gpu_n):
            model.addConstr(T[i] >= t_gpu[i][j] + alpha,
                            name=f"epigraph_{i}_{j}")

    # -------------------------------------------------------------------------
    # Objective (paper Eq. obj): T = sum_i T_i  (each T_i already includes alpha)
    # -------------------------------------------------------------------------
    model.setObjective(
        quicksum(T[i] for i in range(max_iter_n)),
        GRB.MINIMIZE)

    print('Running optimizer (with reuse)...')
    sys.stdout.flush()
    model.optimize(mycallback)

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    assign_list = []
    for i in range(max_iter_n):
        for j in range(gpu_n):
            to_append = []
            for k in range(snapshot_n):
                if x[k][i][j].X > 0.5:
                    to_append.append(k)
            if len(to_append) != 0:
                assign_list.append(to_append)

    imbalance = [0 for _ in range(gpu_n)]
    for idx, snapshot_list in enumerate(assign_list):
        if len(snapshot_list) == 1:
            imbalance[idx % gpu_n] += time_costs[snapshot_list[0]]
        else:
            k1, k2 = snapshot_list[0], snapshot_list[1]
            imbalance[idx % gpu_n] += (time_costs[k1] + time_costs[k2]
                                       - reuse[k1][k2])

    print('Per-GPU load:', imbalance)
    if min(imbalance) > 0:
        print('Imbalance ratio:', max(imbalance) / min(imbalance))
    print('Assignment:', assign_list)
    print(f'Objective T = {model.ObjVal:.2f}'
          f' (= sum_i T_i,  m={max_iter_n}, alpha={alpha})')
    print(f'Runtime = {model.Runtime:.2f}s')

    return model, assign_list



if __name__ == "__main__":
    test_gpus = 8
    alpha = 0.0  # set to measured allreduce time to exactly match paper's T

    time_costs, reuse = read_from_file('YOUR_REUSE_LOG_FILENAME.txt')

    test_snapshots = len(time_costs)
    get_snapshot_alloc_reuse(test_snapshots, test_gpus, time_costs, reuse,
                             L=2, alpha=alpha, ideal_gap=0.05)
