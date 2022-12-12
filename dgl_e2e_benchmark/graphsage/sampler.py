import torch
import gs
import dgl
from dgl import to_block
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
        torch.cuda.nvtx.range_push('dgl graphsage sampling')
        # it=0
        for fanout in reversed(self.fanouts):
            # print("layer:",it)
            # print("memory allocated before dgl sampling: ",torch.cuda.memory_allocated()/(1024*1024*1024))
            # print("memory reserved before dgl sampling: ",torch.cuda.memory_reserved()/(1024*1024*1024))
            torch.cuda.nvtx.range_push('dgl sample neighbors')
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            # print("memory allocated dgl sampling: ",torch.cuda.memory_allocated()/(1024*1024*1024))
            # print("memory reserved dgl sampling: ",torch.cuda.memory_reserved()/(1024*1024*1024))
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push('dgl to block')
            # print("seeds_nodes:",seed_nodes.shape)
            # print("seeds_nodes:",seed_nodes)
            # print("frontier:",frontier)
            # print("frontier nodes:",frontier.nodes())
            block = to_block(frontier, seed_nodes)

            # print("memory allocated dgl to block: ",torch.cuda.memory_allocated()/(1024*1024*1024))
            # print("memory reserved dgl to block: ",torch.cuda.memory_reserved()/(1024*1024*1024))
         #   print("sampled block:",block)
            # print("src node length:",block.srcnodes().shape)
            # print("src nodes:",block.srcnodes())
            # print("dst nodes length:",block.dstnodes().shape)
            # print("dst nodes:",block.dstnodes())
            torch.cuda.nvtx.range_pop()
            seed_nodes = block.srcdata[dgl.NID]
            torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_pop()
            blocks.insert(0, block)
            # it+=1
        # print("after sampling memory allocated:",torch.cuda.memory_allocated()/(1024*1024*1024))
        # print("after sampling memory reserved:",torch.cuda.memory_reserved()/(1024*1024*1024))
        torch.cuda.nvtx.range_pop()
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
        torch.cuda.nvtx.range_push('dgl sampling')
        for fanout in reversed(self.fanouts):
 #           torch.cuda.nvtx.range_push('dgl in subgraph')
            subg = dgl.in_subgraph(g, seed_nodes)
   #         torch.cuda.nvtx.range_pop()
   #         torch.cuda.nvtx.range_push('dgl sample neighbors')
            frontier = subg.sample_neighbors(seed_nodes, fanout)
   #         torch.cuda.nvtx.range_pop()
  #          torch.cuda.nvtx.range_push('dgl to block')
            block = to_block(frontier, seed_nodes)
  #          torch.cuda.nvtx.range_pop()
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        input_nodes = seed_nodes
        torch.cuda.nvtx.range_pop()
        self.sample_time += time.time() - start
        return input_nodes, output_nodes, blocks


def matrix_sampler_nonfused(A: gs.Matrix, seeds, fanouts):
    blocks = []
    output_nodes = seeds
    torch.cuda.nvtx.range_push('nonfused matrix graphsage sampling')
    for fanout in fanouts:
        torch.cuda.nvtx.range_push('matrix slicing')
        subA = A[:, seeds]
        # print("memory allocated after slicing:",torch.cuda.memory_allocated()/(1024*1024*1024))  
        # print("memory reserved after slicing:",torch.cuda.memory_reserved()/(1024*1024*1024))
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('matrix colwise sampling')
        subA = subA.columnwise_sampling(fanout, replace=False)
        torch.cuda.nvtx.range_pop()
        # print("memory allocated after sampling:",torch.cuda.memory_allocated()/(1024*1024*1024)) 
        # print("memory reserved after sampling:",torch.cuda.memory_reserved()/(1024*1024*1024))
        torch.cuda.nvtx.range_push('to dgl block')
        block = subA.to_dgl_block()
        torch.cuda.nvtx.range_pop()
        # print("memory allocated after to_dgl_block:",torch.cuda.memory_allocated()/(1024*1024*1024)) 
        # print("memory reserved after to_dgl_block:",torch.cuda.memory_reserved()/(1024*1024*1024))
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    torch.cuda.nvtx.range_pop()
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def matrix_sampler_fused(A: gs.Matrix, seeds, fanouts):
    blocks = []
    output_nodes = seeds
    torch.cuda.nvtx.range_push('matrix fused graphsage sampling')
    # it = 0
    # print("memory allocated before sampling:",torch.cuda.memory_allocated()/(1024*1024*1024)) 
    # print("memory reserved before sampling",torch.cuda.memory_reserved()/(1024*1024*1024))
    for fanout in fanouts:    
        # print("layer:",it)
        # print("memory allocated before slicing: ",torch.cuda.memory_allocated()/(1024*1024*1024))
        # print("memory reserved before slicing: ",torch.cuda.memory_reserved()/(1024*1024*1024))
        torch.cuda.nvtx.range_push('matrix graphsage slicig and sampling')
        subA = gs.Matrix(A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False))
        torch.cuda.nvtx.range_pop()
        # print("memory allocated after slicing:",torch.cuda.memory_allocated()/(1024*1024*1024))  
        # print("memory reserved after slicing:",torch.cuda.memory_reserved()/(1024*1024*1024))
        torch.cuda.nvtx.range_push('matrix graphsage to dgl block')
        block = subA.to_dgl_block()
        # print("memory allocated after to_dgl_block:",torch.cuda.memory_allocated()/(1024*1024*1024)) 
        # print("memory reserved after to_dgl_block:",torch.cuda.memory_reserved()/(1024*1024*1024))
        seeds = block.srcdata['_ID']
        torch.cuda.nvtx.range_pop()
        blocks.insert(0, block)
        # it+=1
    torch.cuda.nvtx.range_pop()
    # print("memory allocated after sampling:",torch.cuda.memory_allocated()/(1024*1024*1024)) 
    # print("memory reserved after sampling:",torch.cuda.memory_reserved()/(1024*1024*1024))
    input_nodes = seeds
    return input_nodes, output_nodes, blocks
