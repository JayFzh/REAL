# REAL: Efficient Distributed Training of Dynamic GNNs with Reuse-Aware Load Balancing Scheduling

Artifact description for the SC26 submission.

This repository contains the source code backing the four contributions of the
paper, organized one-to-one with the pipeline stages:

| Directory             | Purpose                                                                                                  |
|-----------------------|----------------------------------------------------------------------------------------------------------|
| `data_generate/`      | Downloads raw graphs and builds per-dataset snapshot sequences + DiffMaps.                               |
| `dgl_modification/`   | Patched DGL `GraphConv` and `GATConv` kernels with `IncreConv` and unique-projection entry points.       |
| `ILP_Greedy_solver/`  | Reuse-aware scheduler: REAL-L (Gurobi ILP) and REAL-G (greedy heuristic).                                |
| `training_process/`   | Distributed training driver (`training.py`) plus model, dataloader, and utility modules.                 |

## Quick start

1. Pull the DGL NGC container:
   ```bash
   docker pull nvcr.io/nvidia/dgl:24.09-py3
   ```
2. Inside the container, install extra Python dependencies:
   ```bash
   pip install ogb gurobipy scipy
   ```
3. Apply the DGL kernel patch by copying `dgl_modification/graphconv.py` and
   `dgl_modification/gatconv.py` over the corresponding files in your DGL
   installation (see `dgl_modification/README.MD`).
4. Resolve `YOUR_*` placeholder paths in the scripts (or set the equivalent
   `DGNN_GRAPH_ROOT` / `DGNN_DATA_ROOT` environment variables).
5. Generate datasets and DiffMaps via the scripts in `data_generate/`.
6. Produce a schedule with `ILP_Greedy_solver/ilp_solver_gurobi.py` (REAL-L)
   or `ILP_Greedy_solver/greedy_solver.py` (REAL-G).
7. Launch distributed training:
   ```bash
   python training_process/training.py \
       --dataset arxiv --model wdgcn --world_size 8 \
       --time_window_size 4 --hidden_size 64 \
       --strategy_path <path-to-strategy-file>
   ```

See the AD/AE appendix PDF submitted with the paper for the full reproduction
workflow, hardware requirements, and expected results.
