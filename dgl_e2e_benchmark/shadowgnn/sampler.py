import torch
import dgl
from dgl import to_block, create_block,transforms
from dgl.dataloading import BlockSampler
import time


class ShaDowKHopSampler(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0


    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        for fanout in reversed(self.fanouts):
            torch.cuda.nvtx.range_push('dgl sample_neighbors') 
            frontier = g.sample_neighbors(
                seed_nodes, fanout,
                replace=self.replace, prob=self.prob, exclude_edges=exclude_eids)
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_push('dgl to blocks') 
            block = transforms.to_block(frontier, seed_nodes)
            torch.cuda.nvtx.range_pop() 
            seed_nodes = block.srcdata[dgl.NID]
        torch.cuda.nvtx.range_push('dgl subgraph') 
        subg = g.subgraph(seed_nodes, relabel_nodes=True,store_ids=False)
        torch.cuda.nvtx.range_pop() 
        return seed_nodes, output_nodes, subg


class ShaDowKHopSampler_finegrained(BlockSampler):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        for fanout in reversed(self.fanouts):
            subg = dgl.in_subgraph(g, seed_nodes)
            frontier = subg.sample_neighbors(seed_nodes, fanout)
            block = transforms.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
        subg = g.subgraph(seed_nodes, relabel_nodes=True,store_ids=False)
        return seed_nodes, output_nodes, subg