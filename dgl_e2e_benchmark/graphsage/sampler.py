import torch
import torch.nn.functional as F
import torchmetrics.functional as MF

import dgl
from dgl import to_block, create_block
from dgl.dataloading import BlockSampler
import time



def neighborsampler_dgl(g, seeds, fanout):
    seed_nodes = seeds
    blocks = []
    acc_time = 0
    torch.cuda.synchronize()
    start = time.time()
    for num_pick in fanout:
        sg = dgl.sampling.sample_neighbors(
            g, seed_nodes, num_pick, replace=True)
        block = to_block(sg, seed_nodes)
        block.edata['_ID'] = sg.edata['_ID']
        seed_nodes = block.srcdata['_ID']
        blocks.insert(0, block)
    torch.cuda.synchronize()
    acc_time += time.time() - start
    print(acc_time)
    return blocks

class DGLNeighborSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.sample_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            # torch.cuda.nvtx.range_push('dgl neighbor sampling')
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            block = to_block(frontier, seed_nodes)
            # torch.cuda.nvtx.range_push('all indices')
            seed_nodes = block.srcdata[dgl.NID]
            # torch.cuda.nvtx.range_pop()
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks

class DGLNeighborSampler_finegrained(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.sample_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        start = time.time()
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            subg = dgl.in_subgraph(g, seed_nodes)
            frontier = subg.sample_neighbors(seed_nodes, fanout)
            block = to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        self.sample_time += time.time() - start
        return input_nodes, output_nodes, blocks
