import torch
import gs
from dgl.utils import gather_pinned_tensor_rows
import torch.nn.functional as F
import numpy as np


def w_o_relabel(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        p = subg._CAPI_sum(1, 2, gs._CSR)
        p = p.sqrt()
        row_indices = torch.unique(subg._CAPI_get_coo_rows(False))
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(F.relu(h_u) + 1)

        q = F.normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        subg = subg._CAPI_slicing(selected, 1, gs._CSR, gs._COO, False)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u[idx], h_v, gs._COO)
        W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[idx], gs._COO)
        subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def w_relabel(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, True)
        p = subg._CAPI_sum(1, 2, gs._CSR)
        p = p.sqrt()
        row_indices = subg._CAPI_get_rows()
        num_pick = np.min([row_indices.numel(), fanout])
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(F.relu(h_u) + 1)

        q = F.normalize(p * attention * g_u, p=1.0, dim=0)

        selected = torch.multinomial(q, num_pick, replacement=False)

        subg = subg._CAPI_slicing(selected, 1, gs._CSR, gs._COO, False)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u, h_v, gs._COO)
        W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[selected], gs._COO)
        subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks



def w_o_relabel_selection(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        p = subg._CAPI_sum(1, 2, gs._COO)
        p = p.sqrt()
        row_indices = torch.unique(subg._CAPI_get_coo_rows(False))
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(F.relu(h_u) + 1)

        q = F.normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u[idx], h_v, gs._COO)
        W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[idx], gs._COO)
        subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def w_relabel_selection(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, True)
        p = subg._CAPI_sum(1, 2, gs._COO)
        p = p.sqrt()
        row_indices = subg._CAPI_get_rows()
        num_pick = np.min([row_indices.numel(), fanout])
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(F.relu(h_u) + 1)

        q = F.normalize(p * attention * g_u, p=1.0, dim=0)

        selected = torch.multinomial(q, num_pick, replacement=False)

        subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
        W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u, h_v, gs._COO)
        W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[selected], gs._COO)
        subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks