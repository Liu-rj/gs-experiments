import torch
import load_graph
import time
from sampler import Sampler_FastGCN
import networkx as nx


def fastgcn_sampler(sampler, inds):
    sampled_feats, sampled_adjs, var_loss = sampler.sampling(inds)


device = torch.device("cuda")
dataset = load_graph.load_reddit()
dgl_graph, features = dataset[0], dataset[1]
adj = dgl_graph.adj()
adj = nx.adjacency_matrix(nx.from_edgelist(torch.transpose(adj._indices(), 0, 1)))
print(adj)
input_dim = features.shape[1]
train_nums = adj.shape[0]
layer_sizes = [200, 200]


# sampler = Sampler_FastGCN(None, features, adj,
#                                   input_dim=input_dim,
#                                   layer_sizes=layer_sizes,
#                                   device=device)
