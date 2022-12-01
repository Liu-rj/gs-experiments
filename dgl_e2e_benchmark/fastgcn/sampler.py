import dgl
from dgl.utils import gather_pinned_tensor_rows
import torch
import numpy as np
import gs


class FastGCNSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, probs=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.probs = probs
        self.use_uva = use_uva

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            if self.use_uva:
                node_probs = gather_pinned_tensor_rows(self.probs, nodes)
            else:
                node_probs = self.probs[nodes]
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            subg = dgl.out_subgraph(subg, selected)
            block = dgl.to_block(subg, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def fastgcn_matrix_sampler(A: gs.Matrix, seeds, probs, fanouts):
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        row_indices = subA.row_ids()
        node_probs = probs[row_indices]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, node_probs, fanout, False)
        subA = subA[selected, :]
        block = subA.to_dgl_block()
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, ret
