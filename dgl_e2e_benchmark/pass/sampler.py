import torch
import torch.nn.functional as F
import gs
import dgl
from dgl import to_block
from dgl.dataloading import BlockSampler
from dgl.utils import gather_pinned_tensor_rows


class DGLNeighborSampler(BlockSampler):
    def __init__(self, fanouts, W_1, W_2, sample_a, use_uva, edge_dir='in', features=None):
        super().__init__()
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.W_1 = W_1
        self.W_2 = W_2
        self.sample_a = sample_a
        self.use_uva = use_uva
        self.features = features
        self.ret_loss_tuple = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            subg = dgl.in_subgraph(g, seed_nodes)
            edges = subg.edges()
            nodes = torch.unique(edges[0])

            if self.use_uva:
                u_feats = gather_pinned_tensor_rows(self.features, nodes)
                v_feats = gather_pinned_tensor_rows(self.features, seed_nodes)
            else:
                u_feats = self.features[nodes]
                v_feats = self.features[seed_nodes]
            u_feats_all_w1 = torch.empty(
                (subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device='cuda')
            v_feats_all_w1 = torch.empty(
                (subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device='cuda')
            u_feats_all_w1[nodes] = u_feats @ self.W_1
            v_feats_all_w1[seed_nodes] = v_feats @ self.W_1
            u_feats_all_w2 = torch.empty(
                (subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device='cuda')
            v_feats_all_w2 = torch.empty(
                (subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device='cuda')
            u_feats_all_w2[nodes] = u_feats @ self.W_2
            v_feats_all_w2[seed_nodes] = v_feats @ self.W_2

            att1 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w1,
                                             v_feats_all_w1), dim=1).unsqueeze(1)
            att2 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w2,
                                             v_feats_all_w2), dim=1).unsqueeze(1)
            subg.ndata['v'] = subg.in_degrees()
            subg.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
            att3 = subg.edata['w'].unsqueeze(1)
            att = torch.cat([att1, att2, att3], dim=1)
            att = F.relu(att @ F.softmax(self.sample_a, dim=0))
            att = att + 10e-10 * torch.ones_like(att)

            frontier = dgl.sampling.sample_neighbors(
                subg, seed_nodes, fanout, prob=att, replace=True)
            csc_indptr, csc_indices, edge_ids = frontier.adj_sparse('csc')
            self.ret_loss_tuple = (att[frontier.edata[dgl.EID]][edge_ids],
                                   csc_indices, seed_nodes.numel(), fanout)
            block = to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def matrix_sampler(A: gs.Matrix, seeds, fanouts, features, W_1, W_2, sample_a, use_uva):
    blocks = []
    output_nodes = seeds
    ret_loss_tuple = None
    for fanout in fanouts:
        subA = A[:, seeds]
        neighbors = subA.row_ids()
        subA = subA[neighbors, :]

        if use_uva:
            u_feats = gather_pinned_tensor_rows(features, neighbors)
            v_feats = gather_pinned_tensor_rows(features, seeds)
        else:
            u_feats = features[neighbors]
            v_feats = features[seeds]

        att1 = torch.sum(gs.ops.u_mul_v(subA, u_feats @ W_1,
                         v_feats @ W_1), dim=1).unsqueeze(1)
        att2 = torch.sum(gs.ops.u_mul_v(subA, u_feats @ W_2,
                         v_feats @ W_2), dim=1).unsqueeze(1)
        att3 = subA.normalize(axis=0).get_data().unsqueeze(1)
        att = torch.cat([att1, att2, att3], dim=1)
        att = F.relu(att @ F.softmax(sample_a, dim=0))
        att = att + 10e-10 * torch.ones_like(att)
        subA.set_data(att)

        subA = subA.columnwise_sampling(fanout, replace=True, bias=att)
        ret_loss_tuple = (subA.get_data(order='col'),
                          subA.row_indices(), seeds.numel(), fanout)
        block = subA.to_dgl_block()
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks, ret_loss_tuple


def sampler_loss(loss_tuple, loss_up, features, use_uva):
    # Loss for sampling probability parameters
    # batch_sampler: nodes from upper layer sampled their neighbors
    # batch_sampled: nodes from lower layer were sampled by their parents
    # log probability for "batch_sampler" to sample "batch_sampled"
    data, sampled_nodes, num_cols, fanout = loss_tuple
    logp = torch.log(data).view(num_cols, fanout, -1)
    sampled_feats = gather_pinned_tensor_rows(
        features, sampled_nodes) if use_uva else features[sampled_nodes]
    sampled_feats = sampled_feats.view(num_cols, fanout, -1)
    X = logp * sampled_feats
    X = X.mean(dim=1)
    # Chain rule
    batch_loss = torch.bmm(loss_up.unsqueeze(1), X.unsqueeze(2))
    return batch_loss.mean()
