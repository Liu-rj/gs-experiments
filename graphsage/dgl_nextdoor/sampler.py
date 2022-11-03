import torch
import dgl
from dgl import to_block, create_block
from dgl.dataloading import BlockSampler
import time


torch.ops.load_library("nextdoor/build/libnextdoor.so")


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


def neighborsampler_nextdoor(khop_sampler, seed_nodes: torch.Tensor, fanouts):
    khop_sampler.sample()
    final_samples = khop_sampler.finalSamples()
    khop1_size = fanouts[0] * seed_nodes.shape[0]
    khop2_size = fanouts[1] * khop1_size
    seeds_relabel = torch.arange(
        0, seed_nodes.shape[0], dtype=torch.int64, device='cuda')
    khop1 = final_samples[:khop1_size]
    khop1_relabel = torch.arange(
        0, khop1_size, dtype=torch.int64, device='cuda')
    khop1_dst = seeds_relabel.repeat_interleave(fanouts[0])
    khop2 = final_samples[khop1_size:khop2_size + khop1_size]
    khop2_relabel = torch.arange(
        0, khop2_size, dtype=torch.int64, device='cuda')
    khop2_dst = khop1_relabel.repeat_interleave(fanouts[1])
    blocks = [create_block((khop2_relabel, khop2_dst)),
              create_block((khop1_relabel, khop1_dst))]
    return [seed_nodes, khop1, khop2], blocks


class NextdoorKhopSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None, file_path=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.sampler = torch.classes.my_classes.NextdoorKHopSampler(file_path)
        self.sampler.initSampling()
        # self.sample_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # torch.cuda.synchronize()
        # start = time.time()
        self.sampler.sample()
        final_samples = self.sampler.finalSamples()
        khop1_size = self.fanouts[0] * seed_nodes.shape[0]
        khop2_size = self.fanouts[1] * khop1_size
        seeds_relabel = torch.arange(
            0, seed_nodes.shape[0], dtype=torch.int64, device='cuda')
        khop1 = final_samples[:khop1_size]
        khop1_relabel = torch.arange(
            0, khop1_size, dtype=torch.int64, device='cuda')
        khop1_dst = seeds_relabel.repeat_interleave(self.fanouts[0])
        khop2 = final_samples[khop1_size:khop2_size + khop1_size]
        khop2_relabel = torch.arange(
            0, khop2_size, dtype=torch.int64, device='cuda')
        khop2_dst = khop1_relabel.repeat_interleave(self.fanouts[1])
        blocks = [create_block((khop2_relabel, khop2_dst)),
                  create_block((khop1_relabel, khop1_dst))]
        # torch.cuda.synchronize()
        # self.sample_time += time.time() - start
        return [seed_nodes, khop1, khop2], blocks

    def sample(self, g, seed_nodes, exclude_eids=None):
        return self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)


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
        torch.cuda.synchronize()
        start = time.time()
        # torch.cuda.nvtx.range_push('graphsage sampler func')
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            # torch.cuda.nvtx.range_push('dgl neighbor sampling')
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            blocks.insert(0, (frontier, seed_nodes))
            # torch.cuda.nvtx.range_push('all indices')
            edges = frontier.edges()
            seed_nodes = torch.unique(torch.cat(edges))
            # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        self.sample_time += time.time() - start
        blocks = [to_block(frontier, seed_nodes)
                  for frontier, seed_nodes in blocks]
        return seed_nodes, output_nodes, blocks
