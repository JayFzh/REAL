# DGL Modification

This directory contains modified versions of two DGL convolution layers that
add incremental computation support (IncreConv / IncreGAT) for dynamic graphs.

## Files

| File | Replaces |
|------|----------|
| `graphconv.py` | `dgl/python/dgl/nn/pytorch/conv/graphconv.py` |
| `gatconv.py` | `dgl/python/dgl/nn/pytorch/conv/gatconv.py` |

## How to Apply

Locate your DGL installation path (e.g. via `python -c "import dgl; print(dgl.__file__)"`),
then replace the corresponding files:

```bash
DGL_CONV=$(python -c "import dgl, os; print(os.path.join(os.path.dirname(dgl.__file__), 'nn/pytorch/conv'))")

cp graphconv.py $DGL_CONV/graphconv.py
cp gatconv.py   $DGL_CONV/gatconv.py
```

## Changes vs. Upstream DGL

### `graphconv.py` (`GraphConv`)

- Added optional `reuse` parameter to `__init__`: when `True`, registers a
  separate `reuse_weight` parameter used for the unique-node linear projection.
- `increConv_subgraph(feat, last_tensor, diff_graph)`: incremental GCN
  forward pass on a DiffMap graph. Aggregates only the changed edges
  (weight = +1 for added, −1 for removed) and applies **degree renormalization**
  to correct the destination-side `1/√d` scaling factor as node degrees shift
  between snapshots. Requires `diff_graph.ndata` to contain
  `prev_node_degree` and `curr_node_degree` (populated by
  `data_generate/build_diff_graphs.py`).
- `unique_gemm(unique_feat)`: applies `reuse_weight` to a deduplicated feature
  tensor to avoid redundant matrix multiplications across reused nodes.

### `gatconv.py` (`GATConv`)

- `forward(...)` gains a `reuse` flag: when `True`, returns an extra tuple
  `(el, er, feat_src, node_denom)` for caching attention intermediates.
- `increGAT_forward(graph, reuse_list, last_tensor, change_node)`:
  incremental GAT forward pass. Reuses cached attention logits
  `(el, er, feat_src)` from the previous snapshot and only recomputes the
  softmax denominator correction via `get_numer_with_pm`.
- `edge_softmax_with_denominator`: manual edge-softmax that also returns the
  per-node denominator `∑exp(z)`, enabling incremental updates.
- `normalize_edge_weight`: normalizes per-edge weights by the corresponding
  node denominator.
- `get_numer_with_pm`: computes the signed (`pm = ±1`) weighted sum of
  `exp(logit)` per node, used to update the softmax denominator incrementally.
