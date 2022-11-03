import dgl
import torch
import numba
from numba.core import types
from numba.typed import Dict
import numpy as np
import time


@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai


@numba.jit
def union(*arrays):
    # Faster than np.union1d and torch.unique(torch.cat(...))
    s = set()
    for a in arrays:
        s.update(a)
    a = np.asarray(list(s))
    return a


class LADIESNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, nodes_per_layer, weight='w', out_weight='w', replace=False):
        # super().__init__(len(nodes_per_layer), return_eids=False)
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.epoch_sampling_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        for block_id in reversed(range(len(self.nodes_per_layer))):
            torch.cuda.synchronize()
            start = time.time()

            num_nodes_to_sample = self.nodes_per_layer[block_id]
            insg = dgl.in_subgraph(g, seed_nodes)
            insg = dgl.compact_graphs(insg, seed_nodes)
            cand_nodes = insg.ndata[dgl.NID]
            probs = g.out_degrees().float().cuda()[cand_nodes]

            neighbor_nodes_idx = torch.multinomial(
                probs, num_nodes_to_sample, replacement=False).cpu().numpy()
            seed_nodes_idx = find_indices_in(
                seed_nodes.cpu().numpy(), cand_nodes.cpu().numpy())
            assert seed_nodes_idx.min() != -1
            neighbor_nodes_idx = union(neighbor_nodes_idx, seed_nodes_idx)
            seed_nodes_local_idx = torch.from_numpy(
                find_indices_in(seed_nodes_idx, neighbor_nodes_idx)).cuda()
            assert seed_nodes_idx.min().item() != -1
            neighbor_nodes_idx = torch.from_numpy(neighbor_nodes_idx).cuda()

            sg = insg.subgraph(neighbor_nodes_idx)
            nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID]]

            torch.cuda.synchronize()
            self.epoch_sampling_time += time.time() - start

            block = dgl.to_block(sg, seed_nodes_local_idx)
            # correct node ID mapping
            block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID]]
            block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID]]

            seed_nodes = block.srcdata[dgl.NID]
            block.create_formats_()
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
