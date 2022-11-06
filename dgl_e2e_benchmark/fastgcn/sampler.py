import dgl
from dgl.utils import gather_pinned_tensor_rows
import torch
import numpy as np
import time


class FastGCNSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.use_uva = use_uva
        self.sampling_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        torch.cuda.synchronize()
        start = time.time()
        blocks = []
        output_nodes = seed_nodes
        probs = g.ndata['w']
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            if self.use_uva:
                node_probs = gather_pinned_tensor_rows(probs, nodes)
            else:
                node_probs = probs[nodes]
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            selected = torch.cat((seed_nodes, selected)).unique()
            subg = dgl.out_subgraph(subg, selected)
            block = dgl.to_block(subg, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        torch.cuda.synchronize()
        self.sampling_time += time.time() - start
        return input_nodes, output_nodes, blocks
