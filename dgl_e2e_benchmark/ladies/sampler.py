import dgl
import torch
import time
from dgl.utils import gather_pinned_tensor_rows
import numpy as np


class LADIESSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, weight='w', out_weight='w', replace=False, use_uva=False):
        super().__init__()
        self.fanouts = fanouts
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.sampling_time = 0
        self.use_uva = use_uva

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        torch.cuda.synchronize()
        start = time.time()
        blocks = []
        output_nodes = seed_nodes
        W = g.edata[self.edge_weight]
        all_nodes = g.ndata['nodes']
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            # layer-wise sample
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                weight = gather_pinned_tensor_rows(
                    W, reversed_subg.edata[dgl.EID])
            else:
                weight = W[reversed_subg.edata[dgl.EID]]
            probs = dgl.ops.copy_e_sum(reversed_subg, weight ** 2)
            if self.use_uva:
                nodes = gather_pinned_tensor_rows(
                    all_nodes, (probs > 0).nonzero().flatten())
            else:
                nodes = all_nodes[probs > 0]
            node_probs = probs[probs > 0]
            num_pick = np.min([node_probs.numel(), fanout])
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            ################
            selected = torch.cat((seed_nodes, selected)).unique()
            subg = dgl.out_subgraph(subg, selected)
            weight = weight[subg.edata[dgl.EID]]
            W_tilde = dgl.ops.e_div_u(subg, weight, probs)
            W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
            W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
            block = dgl.to_block(subg, seed_nodes)
            block.edata[self.output_weight] = W_tilde
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        torch.cuda.synchronize()
        self.sampling_time += time.time() - start
        return input_nodes, output_nodes, blocks
