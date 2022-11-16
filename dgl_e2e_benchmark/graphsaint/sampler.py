from model import *
import dgl
from dgl import to_block, create_block,transforms
from dgl.dataloading import BlockSampler
import time

class GraphSaintSampler(object):
    def __init__(self, walk_length=5, use_uva=False):
        super().__init__()
        self.sampling_time = 0
        self.walk_length = walk_length
    def sample(self, g, seeds,exclude_eids=None):
        traces, types = dgl.sampling.random_walk(g, nodes=seeds, length=self.walk_length)
        sampled_nodes = traces.view(traces.numel())
        sampled_nodes = sampled_nodes[sampled_nodes !=-1]
        sampled_nodes = torch.unique(sampled_nodes, sorted=False)
        sg = g.subgraph(sampled_nodes, relabel_nodes=True)
        return sg.ndata['_ID'],sg

class GraphSaintSampler_finegrained(object):
    def __init__(self, walk_length=4, use_uva=False):
        super().__init__()
        self.sampling_time = 0
        self.walk_length = walk_length

    def sample(self, g, seeds,exclude_eids=None):
        ret = [seeds, ]
        for i in range(0, self.walk_length):
            subg= g.in_subgraph(seeds)
            subg = dgl.sampling.sample_neighbors(subg,seeds,1)
            seeds= subg.edges()[0]
            ret.append(seeds)
        sampled_nodes = torch.stack(ret)
        sampled_nodes = sampled_nodes.view(sampled_nodes.numel())
        sampled_nodes = sampled_nodes[sampled_nodes!=-1]
        sampled_nodes = torch.unique(sampled_nodes, sorted=False)
        sg = g.subgraph(sampled_nodes, relabel_nodes=True)
        return sg.ndata['_ID'],sg