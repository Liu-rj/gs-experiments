import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchtext.data.functional import numericalize_tokens_from_iterator
import time


def padding(array, yy, val):
    """
    :param array: torch tensor array
    :param yy: desired width
    :param val: padded value
    :return: padded array
    """
    w = array.shape[0]
    b = 0
    bb = yy - b - w

    return torch.nn.functional.pad(array, pad=(b, bb), mode='constant', value=val)


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.num_nodes(
                self.item_type), (self.batch_size,), device=self.g.device)
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            neg_tails = torch.randint(0, self.g.num_nodes(
                self.item_type), (self.batch_size,), device=self.g.device)

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]


class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                                        random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        torch.cuda.nvtx.range_push("sample blocks")
        blocks = []
        for sampler in self.samplers:
            torch.cuda.nvtx.range_push("dgl pinasage sampler")
            frontier = sampler(seeds)
            torch.cuda.nvtx.range_pop()
            if heads is not None:
                torch.cuda.nvtx.range_push("dgl pinasage remove edge")
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat(
                    [tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    # print(old_frontier)
                    # print(frontier)
                    # print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("to block")
            block = compact_and_copy(frontier, seeds)
            torch.cuda.nvtx.range_pop()
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        torch.cuda.nvtx.range_pop()
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        torch.cuda.nvtx.range_push("sample from item pairs")
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.num_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.num_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        torch.cuda.nvtx.range_pop()
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID].to('cpu')
    #    print("g.device:",g.device)
    #    print("col:",g.nodes[ntype].data[col].device)
   #     print("induced nodes:",induced_nodes)
        ndata[col] = g.nodes[ntype].data[col][induced_nodes].to('cuda')


def assign_textual_node_features(ndata, features):
    node_ids = ndata[dgl.NID].to('cpu')
    for field_name, field in features.items():
        ndata[field_name] = field[node_ids].to('cuda')


def assign_features_to_blocks(blocks, g, features, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, features)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, features)


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

    def collate_test(self, seeds):
        # batch = torch.LongTensor(samples).to(self.g.device)
        blocks = self.sampler.sample_blocks(seeds)
        return blocks
