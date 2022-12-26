import dgl
from dgl.utils import gather_pinned_tensor_rows
import torch
import numpy as np
import gs

_DCSR = 16
_DCSC = 8
_CSR = 4
_CSC = 2
_COO = 1


class LADIESSampler(dgl.dataloading.BlockSampler):

    def __init__(self,
                 fanouts,
                 weight='w',
                 out_weight='w',
                 replace=False,
                 W=None,
                 use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.W = W
        self.use_uva = use_uva

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # torch.cuda.nvtx.range_push('dgl sampler')
        blocks = []
        output_nodes = seed_nodes
        for fanout in self.fanouts:
            # torch.cuda.nvtx.range_push('in subgraph')
            subg = dgl.in_subgraph(g, seed_nodes)
            # torch.cuda.nvtx.range_pop()
            # layer-wise sample
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            # torch.cuda.nvtx.range_push('row sum')
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                weight = gather_pinned_tensor_rows(
                    self.W, reversed_subg.edata[dgl.EID])
            else:
                weight = self.W[reversed_subg.edata[dgl.EID]]
            probs = dgl.ops.copy_e_sum(reversed_subg, weight**2)
            # torch.cuda.nvtx.range_pop()
            node_probs = probs[nodes]
            # torch.cuda.nvtx.range_push('sample')
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            # torch.cuda.nvtx.range_pop()
            selected = nodes[idx]
            ################
            selected = torch.cat((seed_nodes, selected)).unique()
            # torch.cuda.nvtx.range_push('out subgraph')
            subg = dgl.out_subgraph(subg, selected)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('row divide')
            weight = weight[subg.edata[dgl.EID]]
            W_tilde = dgl.ops.e_div_u(subg, weight, probs)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('col normalize')
            W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
            W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
            # torch.cuda.nvtx.range_pop()
            block = dgl.to_block(subg, seed_nodes)
            block.edata[self.output_weight] = W_tilde[block.edata[dgl.EID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        # torch.cuda.nvtx.range_pop()
        return input_nodes, output_nodes, blocks


class LADIESSamplerRelabel(dgl.dataloading.BlockSampler):

    def __init__(self,
                 fanouts,
                 weight='w',
                 out_weight='w',
                 replace=False,
                 W=None,
                 use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.W = W
        self.use_uva = use_uva

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                weight = gather_pinned_tensor_rows(
                    self.W, reversed_subg.edata[dgl.EID])
            else:
                weight = self.W[reversed_subg.edata[dgl.EID]]
            probs = dgl.ops.copy_e_sum(reversed_subg, weight**2)
            node_probs = probs[nodes]
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            relabel_seeds_nodes = torch.ops.gs_ops.index_search(
                subg.ndata[dgl.NID], seed_nodes)
            selected = torch.cat((relabel_seeds_nodes, selected)).unique()
            subg = dgl.out_subgraph(subg, selected, relabel_nodes=False)
            weight = weight[subg.edata[dgl.EID]]
            W_tilde = dgl.ops.e_div_u(subg, weight, probs)
            W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
            W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
            block = dgl.to_block(subg, relabel_seeds_nodes)
            block.edata[self.output_weight] = W_tilde[block.edata[dgl.EID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def ladies_matrix_sampler(P: gs.Matrix, seeds, fanouts):
    # torch.cuda.nvtx.range_push('matrix sampler')
    output_node = seeds
    ret = []
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('col slice')
        U = P[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row sum')
        probs = U.sum(axis=1, powk=2)
        # torch.cuda.nvtx.range_pop()
        row_nodes = U.row_ids()
        node_probs = probs[row_nodes]
        # torch.cuda.nvtx.range_push('sample')
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_nodes, node_probs, fanout, False)
        # torch.cuda.nvtx.range_pop()
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        # torch.cuda.nvtx.range_push('row slice')
        subU = U[nodes, :]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row divide')
        subU = subU.divide(probs[nodes], axis=1)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('col normalize')
        subU = subU.normalize(axis=0)
        # torch.cuda.nvtx.range_pop()
        block = subU.to_dgl_block()
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_node = seeds
    # torch.cuda.nvtx.range_pop()
    return input_node, output_node, ret


def ladies_matrix_sampler_with_format_selection_best(P: gs.Matrix, seeds,
                                                     fanouts):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, _CSC, _COO)
        probs = subg._CAPI_sum(1, 2, _COO)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, _COO, _COO)
        subg = subg._CAPI_divide(probs[nodes], 1, _COO)
        subg = subg._CAPI_normalize(0, _COO)
        block = gs.Matrix(subg).to_dgl_block()
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_node = seeds
    return input_node, output_node, ret


def ladies_matrix_sampler_with_format_selection_coo(P: gs.Matrix, seeds,
                                                    fanouts):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, _CSC, _COO)
        probs = subg._CAPI_sum(1, 2, _CSR)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, _CSR, _COO)
        subg = subg._CAPI_divide(probs[nodes], 1, _CSR)
        subg = subg._CAPI_normalize(0, _CSC)
        block = gs.Matrix(subg).to_dgl_block()
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_node = seeds
    return input_node, output_node, ret


def ladies_matrix_sampler_with_format_selection_coo_full(
        P: gs.Matrix, seeds, fanouts):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_full_slicing(seeds, 0, _CSC)
        probs = subg._CAPI_full_sum(1, 2, _CSR)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_full_slicing(nodes, 1, _CSR)
        subg = subg._CAPI_full_divide(probs, 1, _CSR)
        subg = subg._CAPI_full_normalize(0, _CSC)
        block = gs.Matrix(subg).full_to_dgl_block(seeds)
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_node = seeds
    return input_node, output_node, ret
