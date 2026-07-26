# REAL: Efficient Distributed Training of Dynamic GNNs with Reuse-Aware Load-Balancing Scheduling

*Reuse-aware distributed training for dynamic graph neural networks: incremental
DGL kernels skip computation that is unchanged between snapshots, and ILP /
greedy schedulers place snapshot groups on GPUs to balance both reuse and load.*

> Official implementation of our **SC '26** paper, *REAL: Efficient Distributed
> Training of Dynamic GNNs with Reuse-Aware Load Balancing Scheduling*
> (to appear in the ACM/IEEE International Conference for High Performance
> Computing, Networking, Storage, and Analysis).

REAL is a distributed training framework for dynamic graph neural networks
(DGNNs) on discrete-time dynamic graphs (DTDGs). It exploits the high structural
similarity between consecutive snapshots to skip redundant computation, and
schedules snapshot groups across GPUs so that both the reuse opportunities and
the per-GPU load are balanced.

REAL is built on top of [DGL](https://www.dgl.ai) and PyTorch. It provides two
schedulers — **REAL-L** (an exact Gurobi ILP formulation) and **REAL-G** (a fast
greedy heuristic) — together with patched DGL kernels that perform incremental
aggregation.

## Key features

- **Topology-aware data reuse.** Incremental convolution kernels (`IncreConv`
  for GCN, incremental softmax for GAT) recompute only the edges that change
  between snapshots instead of rerunning full `GraphConv`.
- **Reuse-aware load balancing.** REAL-L (Gurobi ILP) and REAL-G (greedy)
  assign snapshot groups to GPUs to jointly maximize cross-group reuse and
  balance workload.
- **Triple-stream pipeline.** Host-to-device transfer, convolution, and linear
  transformation run on separate CUDA streams to overlap PCIe, HBM, and compute.
- **End-to-end pipeline.** A single script profiles reuse, solves the schedule,
  and launches distributed training.

## Repository structure

| Directory / file        | Purpose                                                                                              |
|-------------------------|------------------------------------------------------------------------------------------------------|
| `data_generate/`        | Downloads raw graphs and builds per-dataset snapshot sequences and DiffMaps.                         |
| `dgl_modification/`     | Patched DGL `GraphConv` and `GATConv` kernels with incremental (`IncreConv`) and unique-projection entry points. See [`dgl_modification/README.MD`](dgl_modification/README.MD). |
| `ILP_Greedy_solver/`    | Reuse-aware schedulers: `ilp_solver_gurobi.py` (REAL-L) and `greedy_solver.py` (REAL-G).             |
| `training_process/`     | Distributed training driver (`training.py`), model definitions, dataloader, profiler, and utilities. |
| `build_strategy.py`     | Turns a profiled reuse log into a GPU assignment / schedule file for the training driver.            |
| `run_pipeline.sh`       | End-to-end pipeline: profile → build strategy → train.                                               |

## Requirements

- NVIDIA GPU(s) with CUDA 12.5
- DGL 2.4 and PyTorch 2.4 (the NGC container below bundles both)
- Python packages: `ogb`, `scipy`, and `gurobipy` (only needed for REAL-L)
- A Gurobi license for REAL-L (a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) works). REAL-G has no such dependency.

## Installation

1. Pull the DGL NGC container:
   ```bash
   docker pull nvcr.io/nvidia/dgl:24.09-py3
   ```
2. Inside the container, install the extra Python dependencies:
   ```bash
   pip install ogb gurobipy scipy
   ```
3. Apply the incremental DGL kernels by copying the patched files over your DGL
   installation (details in [`dgl_modification/README.MD`](dgl_modification/README.MD)):
   ```bash
   DGL_CONV=$(python -c "import dgl, os; print(os.path.join(os.path.dirname(dgl.__file__), 'nn/pytorch/conv'))")
   cp dgl_modification/graphconv.py "$DGL_CONV/graphconv.py"
   cp dgl_modification/gatconv.py   "$DGL_CONV/gatconv.py"
   ```

## Datasets

Snapshot sequences and DiffMaps are built by the scripts in `data_generate/`.
Supported datasets: `arxiv`, `products`, `reddit`, `stackoverflow`,
`stackoverflow-a2q`, and `wiki`. Point the scripts at your data location via the
`DGNN_GRAPH_ROOT` / `DGNN_DATA_ROOT` environment variables (or edit the
`YOUR_*` placeholder paths in the scripts).

## Usage

### End-to-end pipeline

The quickest way to run everything (profile → schedule → train) is:

```bash
DGNN_GRAPH_ROOT=/path/to/graphs DGNN_DATA_ROOT=/path/to/data \
DATASET=arxiv MODEL=wdgcn SOLVER=greedy GPUS=8 WORLD_SIZE=8 \
./run_pipeline.sh
```

All knobs (`DATASET`, `MODEL`, `WINDOW`, `NUM_SNAPSHOTS`, `SOLVER`, `GPUS`,
`WORLD_SIZE`, `HIDDEN`, `EPOCHS`, ...) can be overridden through environment
variables; see the top of [`run_pipeline.sh`](run_pipeline.sh).

### Step by step

1. **Generate datasets and DiffMaps** with the scripts in `data_generate/`.
2. **Build a schedule** from a profiled reuse log:
   ```bash
   python build_strategy.py --reuse-log reuse.txt --solver greedy \
       --gpus 8 --world-size 8 --model wdgcn --out strategy.txt
   ```
   Use `--solver ilp` for REAL-L (requires `gurobipy`).
3. **Launch distributed training:**
   ```bash
   python training_process/training.py \
       --dataset arxiv --model wdgcn --world_size 8 \
       --time_window_size 4 --hidden_size 64 \
       --strategy_path strategy.txt
   ```

### Options

- **Datasets** (`--dataset`): `arxiv`, `products`, `reddit`, `stackoverflow`, `stackoverflow-a2q`, `wiki`
- **Models** (`--model`): `evolvegcn`, `wdgcn`, `tgcn`, `gatlstm`
- `--num_layers`: number of GNN layers; reuse applies to the first layer, higher layers recompute in full.
- `--no_3stream`: disable the triple-stream pipeline (intra-snapshot reuse only).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fan2026real,
  title     = {REAL: Efficient Distributed Training of Dynamic GNNs with
               Reuse-Aware Load Balancing Scheduling},
  author    = {Fan, Zihao and Liu, Yunzhuo and Guo, Tian and Jiang, Bo},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage, and Analysis (SC '26)},
  year      = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
