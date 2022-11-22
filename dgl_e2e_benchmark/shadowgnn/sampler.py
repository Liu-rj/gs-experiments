import torch
import dgl
from dgl import to_block, create_block,transforms
from dgl.dataloading import BlockSampler
import time
import gs
from gs.utils import SeedGenerator,ConvModel
def create_dgl_graph(matrix):
    unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = matrix._graph._CAPI_relabel(
    )
    graph = dgl.graph(
        ('csc', (format_tensor1, format_tensor2, [])),
        device='cuda')
    return graph
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
        #    torch.cuda.nvtx.range_push('dgl sample_neighbors') 
            frontier = g.sample_neighbors(
                seed_nodes, fanout,
                replace=self.replace, prob=self.prob, exclude_edges=exclude_eids)
        #    torch.cuda.nvtx.range_pop() 
        #    torch.cuda.nvtx.range_push('dgl to blocks') 
            block = transforms.to_block(frontier, seed_nodes)
        #    torch.cuda.nvtx.range_pop() 
            seed_nodes = block.srcdata[dgl.NID]
        #torch.cuda.nvtx.range_push('dgl subgraph') 
        subg = g.subgraph(seed_nodes, relabel_nodes=True,store_ids=False)
        #torch.cuda.nvtx.range_pop() 
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

class ShaDowKHopSampler_matrix(BlockSampler):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def shadowgnn_matrix_sampler(A: gs.Matrix,seeds):
        output_nodes = seeds
        for fanout in reversed(self.fanouts):
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            seeds = subA.all_indices()
        retA = A[seeds, seeds]
        graph = create_dgl_graph(retA)
        return  retA,seeds

class ShaDowKHopSampler_matrix(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        for fanout in reversed(self.fanouts):
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            seeds = subA.all_indices()
        retA = A[seeds, seeds]
        graph = create_dgl_graph(retA)
        return  seeds,graph

class ShaDowKHopSampler_nonfused(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        #print("seed device:",seeds.device)
        for fanout in reversed(self.fanouts):
            subA = A[:, seeds]
            subA = subA.columnwise_sampling(fanout, False)
            seeds = subA.all_indices()
        # print("seeds device:",seeds.device)
        # print("graph device:",A._graph._CAPI_metadata())
        retA = A[:, seeds]
        retA = retA[seeds,:]
        graph = create_dgl_graph(retA)
        return  seeds,graph