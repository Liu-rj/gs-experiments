import torch
import dgl
from dgl import to_block, create_block,transforms
from dgl.dataloading import BlockSampler
import time
import gs
from gs.utils import SeedGenerator,ConvModel
COO = 1
CSC = 2
CSR = 4
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
        torch.cuda.nvtx.range_push("dgl shadow sampling")
        for fanout in reversed(self.fanouts):
            torch.cuda.nvtx.range_push('dgl sample_neighbors') 
            frontier = g.sample_neighbors(
                seed_nodes, fanout,
                replace=self.replace, prob=self.prob, exclude_edges=exclude_eids)
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_push('dgl to blocks') 
            block = transforms.to_block(frontier, seed_nodes)
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_push('dgl get src id') 
            seed_nodes = block.srcdata[dgl.NID]
            torch.cuda.nvtx.range_pop() 
        torch.cuda.nvtx.range_push('dgl subgraph') 
        subg = g.subgraph(seed_nodes, relabel_nodes=True,store_ids=False)
        torch.cuda.nvtx.range_pop() 
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


class ShaDowKHopSampler_matrix_fusedv1(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        torch.cuda.nvtx.range_push("shadow sampling fusedv1")
        for fanout in reversed(self.fanouts):
            torch.cuda.nvtx.range_push("columnwise slicing and sampling")
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("matrix nodesubgraph v1")
        retA = A[seeds, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("matrix create dgl graph")
        graph = create_dgl_graph(retA)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return  seeds,graph

class ShaDowKHopSampler_matrix_fusedv2(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        torch.cuda.nvtx.range_push("shadow sampling fusedv2")
        for fanout in reversed(self.fanouts):
            torch.cuda.nvtx.range_push("columnwise slicing and sampling")
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("matrix nodesubgraph v2")
        retA =  gs.Matrix(A._graph._CAPI_fusion_slicing(seeds))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("matrix create dgl graph")
        graph = create_dgl_graph(retA)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return  seeds,graph

# class ShaDowKHopSampler_nonfused(object):
#     def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
#         super().__init__()
#         self.fanouts = fanouts
#         self.replace = replace
#         self.prob = prob
#         self.sampling_time = 0
#     def sample(self,A: gs.Matrix,seeds):
#         output_nodes = seeds
#         torch.cuda.nvtx.range_push("shadow sampling nonfused")
#         for fanout in reversed(self.fanouts):
#             torch.cuda.nvtx.range_push("columnwise slicing")
#             subA = A[:, seeds]
#             torch.cuda.nvtx.range_pop()
#             torch.cuda.nvtx.range_push("columnwise sampling")
#             subA = subA.columnwise_sampling(fanout, False)
#             torch.cuda.nvtx.range_pop()
#             torch.cuda.nvtx.range_push("all_indices")
#             seeds = subA.all_indices()
#             torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push("matrix nodesubgraph")
#         retA = A[seeds,seeds]
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push("matrix create dgl graph")
#         graph = create_dgl_graph(retA)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_pop()
#         return  seeds,graph

class ShaDowKHopSampler_with_format_selection_coo(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        for fanout in reversed(self.fanouts):
            subg =  A._graph._CAPI_slicing(seeds, 0,CSC, COO);
            subg = subg._CAPI_sampling(0,fanout,False,CSC,COO)
            seeds = subg._CAPI_all_valid_node()
        subg = A._graph._CAPI_slicing(seeds, 0,CSC, COO);
        subg = subg._CAPI_slicing(seeds, 1,CSC, COO);
        graph = create_dgl_graph(gs.Matrix(subg))
        return  seeds,graph

class ShaDowKHopSampler_with_format_selection_best(object):
    def __init__(self, fanouts, replace=False, prob=None, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.sampling_time = 0
    def sample(self,A: gs.Matrix,seeds):
        output_nodes = seeds
        for fanout in reversed(self.fanouts):
            subg =   A._graph._CAPI_slicing(seeds, 0, CSC, CSC);
            subg = subg._CAPI_sampling(0,fanout,False,CSC,CSC)
            seeds = subg._CAPI_all_valid_node()
        subg = A._graph._CAPI_slicing(seeds, 0,CSC, CSC);
        subg = subg._CAPI_slicing(seeds, 1,CSC, CSC);
        graph = create_dgl_graph(gs.Matrix(subg))
        return  seeds,graph