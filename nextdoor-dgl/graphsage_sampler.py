import torch
import dgl
from dgl import to_block, create_block
import numpy as np
from dgl.dataloading import BlockSampler
import time


def neighborsampler_dgl(g, seeds, fanout):
    seed_nodes = seeds
    blocks = []
    for num_pick in fanout:
        sg = dgl.sampling.sample_neighbors(
            g, seed_nodes, num_pick, replace=True)
        block = to_block(sg, seed_nodes)
        block.edata['_ID'] = sg.edata['_ID']
        seed_nodes = block.srcdata['_ID']
        blocks.insert(0, block)
    return blocks


def neighborsampler_nextdoor(NextDoorKHopSampler, lib, seeds: torch.Tensor, fanout):
    NextDoorKHopSampler.sample()
    finalSamples = np.asarray(lib.finalSamplesArray())
    khop1_size = fanout[0] * seeds.shape[0]
    khop2_size = fanout[1] * khop1_size
    seeds_relabel = torch.arange(0, seeds.shape[0], dtype=torch.int64, device='cuda')
    khop1 = torch.tensor(finalSamples[:khop1_size], dtype=torch.int64, device='cuda')
    khop1_relabel = torch.arange(0, khop1_size, dtype=torch.int64, device='cuda')
    khop1_dst = seeds_relabel.repeat_interleave(fanout[0])
    khop2 = torch.tensor(finalSamples[khop1_size:khop2_size + khop1_size], dtype=torch.int64, device='cuda')
    khop2_relabel = torch.arange(0, khop2_size, dtype=torch.int64, device='cuda')
    khop2_dst = khop1_relabel.repeat_interleave(fanout[1])
    blocks = [create_block((khop2_relabel, khop2_dst)), create_block((khop1_relabel, khop1_dst))]
    return [seeds, khop1, khop2], blocks


class NextdoorKhopSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None, sampler=None, lib=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.sampler = sampler
        self.lib = lib
        # self.sample_time = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # torch.cuda.synchronize()
        # start = time.time()
        self.sampler.sample()
        final_samples = np.asarray(self.lib.finalSamplesArray())
        khop1_size = self.fanouts[0] * seed_nodes.shape[0]
        khop2_size = self.fanouts[1] * khop1_size
        seeds_relabel = torch.arange(0, seed_nodes.shape[0], dtype=torch.int64, device='cuda')
        khop1 = torch.tensor(final_samples[:khop1_size], dtype=torch.int64, device='cuda')
        khop1_relabel = torch.arange(0, khop1_size, dtype=torch.int64, device='cuda')
        khop1_dst = seeds_relabel.repeat_interleave(self.fanouts[0])
        khop2 = torch.tensor(final_samples[khop1_size:khop2_size + khop1_size], dtype=torch.int64, device='cuda')
        khop2_relabel = torch.arange(0, khop2_size, dtype=torch.int64, device='cuda')
        khop2_dst = khop1_relabel.repeat_interleave(self.fanouts[1])
        blocks = [create_block((khop2_relabel, khop2_dst)), create_block((khop1_relabel, khop1_dst))]
        # torch.cuda.synchronize()
        # self.sample_time += time.time() - start
        return [seed_nodes, khop1, khop2], blocks
    
    def sample(self, g, seed_nodes, exclude_eids=None):
        return self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
