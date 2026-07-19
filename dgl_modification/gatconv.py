"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair
from ...functional import edge_softmax
from ..utils import Identity


# pylint: enable=W0235
class GATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False,reuse=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # edge softmax; also returns the per-node denominator and max logit
            # so the incremental path can update attention across snapshots.
            res, node_denom, node_max = self.edge_softmax_with_denominator(graph,e)
            graph.edata["a"] = self.attn_drop(res)
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)
            
            if reuse:
                return rst, (el.clone().detach(), er.clone().detach(), feat_src.clone().detach(),node_denom.clone().detach(), node_max.clone().detach())

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
    

    def increGAT_forward(self,graph,reuse_list,last_tensor,change_node):
        diff_value = graph.edata['weight']
        (el,er,feat_src,old_node_denom,old_node_max) = reuse_list
        graph.srcdata.update({"ft": feat_src, "el": el})
        graph.dstdata.update({"er": er})
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        e = self.leaky_relu(graph.edata.pop("e"))

        # Shift the softmax by the per-node max logit. The shift covers both the
        # cached edges and the newly added edges, so it is the larger of the
        # cached per-node max and the max logit over the added edges.
        _, dst = graph.edges(order="eid")
        N = graph.num_nodes()
        extra = e.shape[1:]
        idx = dst.to(e.device).view((-1,) + (1,) * len(extra)).expand_as(e)
        added = (diff_value > 0).view((-1,) + (1,) * (e.dim() - diff_value.dim())
                                      ).expand_as(e) if diff_value.dim() < e.dim() \
                else (diff_value > 0)
        e_added = e.masked_fill(~added, float("-inf"))
        added_max = e.new_full((N,) + extra, float("-inf")).scatter_reduce(
            0, idx, e_added, reduce="amax", include_self=True)
        node_max = th.maximum(old_node_max, added_max)
        # Re-express the cached denominator under the updated shift.
        denom_rescaled = old_node_denom * th.exp(old_node_max - node_max)

        # Incremental denominator change and per-edge weights under the same shift.
        incre_node_demon, edge_weight_no_norm = self.get_numer_with_pm(
            graph, e, diff_value, node_max=node_max)
        node_denom = denom_rescaled + incre_node_demon
        # Keep the denominator strictly positive before normalizing.
        eps = 1e-12
        safe_denom = th.where(node_denom > eps, node_denom,
                              denom_rescaled.clamp_min(eps))
        # Rescale the cached aggregation by the denominator ratio and add the
        # contribution of the changed edges.
        last_tensor = last_tensor * denom_rescaled / safe_denom
        graph.edata["a"] = self.attn_drop(self.normalize_edge_weight(graph, edge_weight_no_norm, safe_denom, norm_by="dst"))
        graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
        rst = last_tensor + graph.dstdata["ft"]
        return rst, (el,er,feat_src,safe_denom,node_max)
    

    def edge_softmax_with_denominator(self, graph,
                                    logits,
                                    norm_by = "dst",
                                    eps = 1e-12):
        """
        Compute the edge softmax and return the per-edge weights, the per-node
        denominator (sum of exp), and the per-node max logit.

        The softmax is computed as exp(z - m_j) / sum_k exp(z_k - m_j), where m_j
        is the per-node max logit (grouped by ``norm_by``). The max logit is
        returned so the incremental path can update the denominator across
        snapshots under a consistent shift.

        Args:
            graph:   DGLGraph (homogeneous)
            logits:  Tensor (E, *extra), per-edge score z_ij;
                     *extra can be the multi-head dimension (H,)
            norm_by: 'dst' (group by destination node's in-edges, GAT default)
                     or 'src' (group by source node's out-edges)
            eps:     small constant to prevent division by zero

        Returns:
            edge_alpha: (E, *extra), softmax weight for each edge
            node_denom: (N, *extra), per-node denominator sum exp(z - m)
            node_max:   (N, *extra), per-node max logit used as the shift
        """
        assert norm_by in ("dst", "src"), "norm_by must be 'dst' or 'src'"

        E = graph.num_edges()
        src, dst = graph.edges(order="eid")  # aligned with logits order
        index = dst if norm_by == "dst" else src
        N = graph.num_nodes()
        extra_shape = logits.shape[1:]

        # Per-node max logit used as the softmax shift. Detached: a constant
        # shift leaves the softmax and its gradient unchanged.
        idx_exp = index.view((-1,) + (1,) * len(extra_shape)).expand_as(logits)
        node_max = logits.new_full((N,) + extra_shape, float("-inf"))
        node_max = node_max.scatter_reduce(0, idx_exp, logits.detach(),
                                           reduce="amax", include_self=True)
        # Nodes with no in-edges keep -inf; use a zero shift for them.
        node_max = th.nan_to_num(node_max, neginf=0.0)

        # exp(z - m_j)
        max_on_edges = node_max.index_select(0, index)
        exp_logits = th.exp(logits - max_on_edges)

        # Denominator sum_k exp(z_k - m_j) per node
        node_denom = exp_logits.new_zeros((N,) + extra_shape)
        node_denom.index_add_(0, index, exp_logits)

        # Gather per-edge denominators from node_denom
        denom_on_edges = node_denom.index_select(0, index)

        # softmax = exp(z - m) / sum exp(z - m)   (shift m cancels)
        edge_alpha = exp_logits / (denom_on_edges + eps)

        return edge_alpha, node_denom, node_max
    
    def normalize_edge_weight(self, graph, edge_weight, node_denom, norm_by="dst", eps=1e-12):
        """
        Normalize per-edge weights by the corresponding node denominator:
            normalized_edge = edge_weight / node_denom[node_index]

        Args:
            graph:       DGLGraph
            edge_weight: (E, H, 1), weight for each edge
            node_denom:  (N, H, 1), per-node denominator
            norm_by:     'dst' (normalize by destination node denominator, GAT default)
                         or 'src' (normalize by source node denominator)
            eps:         small constant to prevent division by zero

        Returns:
            normalized_edge: (E, H, 1), normalized edge weights
        """
        assert norm_by in ("dst", "src"), "norm_by must be 'dst' or 'src'"

        # Retrieve edge endpoints aligned with edge_weight ordering
        src, dst = graph.edges(order="eid")
        index = dst if norm_by == "dst" else src

        # Gather the denominator for each edge from node_denom
        denom_on_edges = node_denom.index_select(0, index)

        normalized_edge = edge_weight / (denom_on_edges + eps)

        return normalized_edge

    def get_numer_with_pm(self, graph, logits, pm, node_max=None, norm_by="dst"):
        """
        Compute per-node weighted aggregation and return per-edge weights.
        For each edge (i->j): edge_weight = exp(logit - m_j) * pm,
        then sum edge_weight grouped by node (dst or src).

        ``node_max`` (m_j) is the per-node softmax shift from the base snapshot.
        Applying the same shift used to build the cached denominator keeps the
        incremental update consistent. If ``node_max`` is None, no shift is
        applied.

        Args:
            graph:    DGLGraph
            logits:   (E, *extra), per-edge score values
            pm:       (E,) or (E, *extra), elements are +1 or -1
            node_max: (N, *extra) per-node shift, or None
            norm_by:  'dst' (group by destination node, default)
                      or 'src' (group by source node)

        Returns:
            node_value:  (N, *extra), accumulated result per node
            edge_weight: (E, *extra), per-edge weight (= pm * exp(logit - m))
        """
        assert norm_by in ("dst", "src"), "norm_by must be 'dst' or 'src'"

        # Retrieve edge endpoints in eid order, aligned with logits dim-0
        src, dst = graph.edges(order="eid")
        index = dst if norm_by == "dst" else src
        N = graph.num_nodes()

        # Broadcast pm to match the trailing dimensions of logits
        if pm.dim() == 1:
            # (E,) -> (E, 1, 1, ...) then expand to logits.shape
            expand_shape = (pm.shape[0],) + (1,) * (logits.dim() - 1)
            pm_expanded = pm.view(*expand_shape).expand_as(logits)
        else:
            if pm.shape != logits.shape:
                raise ValueError(
                    f"pm shape {pm.shape} must be (E,) or match logits shape {logits.shape}"
                )
            pm_expanded = pm

        # Ensure device alignment
        device = logits.device
        pm_expanded = pm_expanded.to(device)
        index = index.to(device)

        # Apply the per-node softmax shift used by the cached denominator.
        if node_max is not None:
            logits = logits - node_max.index_select(0, index)

        # Per-edge weight: pm * exp(logit - m), same shape as logits
        edge_weight = th.exp(logits) * pm_expanded

        # Initialize per-node accumulation buffer: (N, *extra)
        extra_shape = logits.shape[1:]
        node_value = edge_weight.new_zeros((N,) + extra_shape)

        # Scatter-add each edge's value into the corresponding node bucket
        node_value.index_add_(0, index, edge_weight)

        return node_value, edge_weight