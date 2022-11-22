from model import *
import dgl
from dgl import to_block, create_block,transforms
from dgl.dataloading import BlockSampler
import time
import gs

def create_dgl_graph(matrix):
    unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = matrix._graph._CAPI_relabel(
    )
    graph = dgl.graph(
        ('csc', (format_tensor1, format_tensor2, [])),
        device='cuda')
    return graph

class GraphSaintSampler(object):
    def __init__(self, walk_length=4, use_uva=False):
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
class GraphSaintSampler_matrix(object):
    def __init__(self, walk_length=4, use_uva=False):
        super().__init__()
        self.sampling_time = 0
        self.walk_length = walk_length
    def sample(self,A: gs.Matrix, seeds):
        paths = A.random_walk(seeds, self.walk_length)
        node_ids = paths.view(paths.numel())
        node_ids = node_ids[node_ids!=-1]
        out = torch.unique(node_ids, sorted=False)
        induced_subA = A[out, out]
        subA = create_dgl_graph(induced_subA)
        return out,subA
class GraphSaintSampler_matrix_nonfused(object):
    def __init__(self, walk_length=4, use_uva=False):
        super().__init__()
        self.sampling_time = 0
        self.walk_length = walk_length
    def sample(self,A: gs.Matrix, seeds):
        ret = [seeds, ]
        for i in range(0, self.walk_length):
            subA = A[:,seeds]
            subA = subA.columnwise_sampling(1,True)
            seeds = subA.row_indices()
            ret.append(seeds)
        node_ids = torch.stack(ret)
        node_ids = node_ids.view(node_ids.numel())
        node_ids = node_ids[node_ids != -1]
        out = torch.unique(node_ids, sorted=False)
        induced_subA = A[out, out]
        subA = create_dgl_graph(induced_subA)
        return out,subA

