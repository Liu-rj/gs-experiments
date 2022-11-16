import dgl
from dgl.utils import gather_pinned_tensor_rows
import torch
from torch.nn.functional import normalize, relu
import numpy as np
import gs


class FastGCNSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, use_uva=False, W=None, eweight=None):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.use_uva = use_uva
        self.W = W
        self.edge_weight = eweight

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        features = g.ndata['feat']
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            reversed_subg = dgl.reverse(subg, copy_edata=True)
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
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subA = A[:, seeds]
        p = subA.sum(axis=0, powk=2).sqrt()
        row_indices = subA.row_ids()
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

        subA = subA[selected, :]
        subA = subA.multiply(h_v, axis=0).multiply(h_u, axis=1)
        subA = subA.divide()
        block = subA.to_dgl_block()
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks
