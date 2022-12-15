import dgl
from dgl.utils import gather_pinned_tensor_rows
import torch
from torch.nn.functional import normalize, relu
import numpy as np
import gs

_DCSR = 16
_DCSC = 8
_CSR = 4
_CSC = 2
_COO = 1


class ASGCNSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, use_uva=False, W=None, eweight=None, node_feats=None):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.use_uva = use_uva
        self.W = W
        self.edge_weight = eweight
        self.node_feats = node_feats

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        features = self.node_feats
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                sampled_e_weight = gather_pinned_tensor_rows(
                    self.edge_weight, reversed_subg.edata[dgl.EID])
            else:
                sampled_e_weight = self.edge_weight[reversed_subg.edata[dgl.EID]]
            p = torch.sqrt(dgl.ops.copy_e_sum(
                reversed_subg, sampled_e_weight ** 2))

            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.numel(), fanout])
            if self.use_uva:
                node_feats_u = gather_pinned_tensor_rows(features, nodes)
                node_feats_v = gather_pinned_tensor_rows(features, seed_nodes)
            else:
                node_feats_u = features[nodes]
                node_feats_v = features[seed_nodes]
            h_u = node_feats_u @ self.W[:, 0]
            h_v = node_feats_v @ self.W[:, 1]
            h_v_sum = torch.sum(h_v)
            attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
            g_u = torch.flatten(relu(h_u) + 1)

            q = normalize(p[nodes] * attention * g_u, p=1.0, dim=0)

            idx = torch.multinomial(q, num_pick, replacement=False)
            selected = nodes[idx]
            subg = dgl.out_subgraph(subg, selected)

            q_allnodes = torch.empty(
                subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_u_allnodes = torch.empty(
                subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_v_allnodes = torch.empty(
                subg.num_nodes(), dtype=torch.float32, device=subg.device)
            q_allnodes[selected] = q[idx]
            h_u_allnodes[selected] = h_u[idx]
            h_v_allnodes[seed_nodes] = h_v

            W_tilde = dgl.ops.u_add_v(
                subg, h_u_allnodes, h_v_allnodes)
            W_tilde = (relu(W_tilde) + 1) / num_pick
            W_tilde = dgl.ops.e_div_u(subg, W_tilde, q_allnodes)
            W_tilde = W_tilde * sampled_e_weight[subg.edata[dgl.EID]]
            # reversed copy_e_sum
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            u_sum = dgl.ops.copy_e_sum(reversed_subg, W_tilde)

            block = dgl.to_block(subg, seed_nodes)
            block.edata['w'] = W_tilde[block.edata[dgl.EID]]
            block.srcdata['u_sum'] = u_sum[block.srcdata[dgl.NID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def asgcn_matrix_sampler(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    # torch.cuda.nvtx.range_push('matrix sampler')
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('col slice')
        subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row sum')
        p = subA.sum(axis=1, powk=2)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sqrt')
        p = p.sqrt()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('valid row id')
        row_indices = subA.row_ids()
        # torch.cuda.nvtx.range_pop()
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(relu(h_u) + 1)

        q = normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        # torch.cuda.nvtx.range_push('sample')
        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)
        # torch.cuda.nvtx.range_pop()

        # torch.cuda.nvtx.range_push('row slice')
        subA = subA[selected, :]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('u add v')
        W_tilde = gs.ops.u_add_v(subA, h_u[idx], h_v)
        # torch.cuda.nvtx.range_pop()
        W_tilde = (relu(W_tilde) + 1) / selected.numel()
        # torch.cuda.nvtx.range_push('e div u')
        W_tilde = gs.ops.e_div_u(subA, W_tilde, q[idx])
        # torch.cuda.nvtx.range_pop()
        subA.set_data(W_tilde * subA.get_data())
        # torch.cuda.nvtx.range_push('row sum')
        u_sum = subA.sum(axis=1)
        # torch.cuda.nvtx.range_pop()
        u_all = torch.zeros(
            A.get_num_rows(), dtype=torch.float32, device='cuda')
        u_all[selected] = u_sum

        block = subA.to_dgl_block()
        block.srcdata['u_sum'] = u_all[block.srcdata['_ID']]
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    # torch.cuda.nvtx.range_pop()
    return input_nodes, output_nodes, blocks


def asgcn_matrix_sampler_with_format_selection_best(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, _CSC, _COO)
        p = subg._CAPI_sum(1, 2, _COO)
        p = p.sqrt()
        row_indices = subg._CAPI_get_valid_rows()
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(relu(h_u) + 1)

        q = normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        subg = subg._CAPI_slicing(selected, 1, _COO, _COO)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u[idx], h_v, _COO)
        W_tilde = (relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[idx], _COO)
        subg._CAPI_set_data(
            W_tilde * subg._CAPI_get_data('default'), 'default')
        u_sum = subg._CAPI_sum(1, 1, _COO)
        u_all = torch.zeros(
            A.get_num_rows(), dtype=torch.float32, device='cuda')
        u_all[selected] = u_sum

        block = gs.Matrix(subg).to_dgl_block()
        block.srcdata['u_sum'] = u_all[block.srcdata['_ID']]
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def asgcn_matrix_sampler_with_format_selection_coo(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, _CSC, _COO)
        p = subg._CAPI_sum(1, 2, _CSR)
        p = p.sqrt()
        row_indices = subg._CAPI_get_valid_rows()
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(relu(h_u) + 1)

        q = normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        subg = subg._CAPI_slicing(selected, 1, _CSR, _COO)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u[idx], h_v, _COO)
        W_tilde = (relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[idx], _CSR)
        subg._CAPI_set_data(
            W_tilde * subg._CAPI_get_data('default'), 'default')
        u_sum = subg._CAPI_sum(1, 1, _CSR)
        u_all = torch.zeros(
            A.get_num_rows(), dtype=torch.float32, device='cuda')
        u_all[selected] = u_sum

        block = gs.Matrix(subg).to_dgl_block()
        block.srcdata['u_sum'] = u_all[block.srcdata['_ID']]
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def asgcn_matrix_sampler_with_format_selection_coo_full(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_full_slicing(seeds, 0, _CSC)
        p = subg._CAPI_full_sum(1, 2, _CSR)
        p = p.sqrt()
        row_indices = subg._CAPI_get_valid_rows()
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(relu(h_u) + 1)

        q = normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        q_allnodes = torch.empty(
            subg._CAPI_full_get_num_nodes(), dtype=torch.float32, device='cuda')
        h_u_allnodes = torch.empty(
            subg._CAPI_full_get_num_nodes(), dtype=torch.float32, device='cuda')
        h_v_allnodes = torch.empty(
            subg._CAPI_full_get_num_nodes(), dtype=torch.float32, device='cuda')
        q_allnodes[selected] = q[idx]
        h_u_allnodes[selected] = h_u[idx]
        h_v_allnodes[seeds] = h_v

        subg = subg._CAPI_full_slicing(selected, 1, _CSR)
        W_tilde = gs.ops.u_add_v(
            gs.Matrix(subg), h_u_allnodes, h_v_allnodes, _COO)
        W_tilde = (relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q_allnodes, _CSR)
        subg._CAPI_set_data(
            W_tilde * subg._CAPI_get_data('default'), 'default')
        u_sum_all = subg._CAPI_sum(1, 1, _CSR)

        block = gs.Matrix(subg).full_to_dgl_block(seeds)
        block.srcdata['u_sum'] = u_sum_all[block.srcdata['_ID']]
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks
