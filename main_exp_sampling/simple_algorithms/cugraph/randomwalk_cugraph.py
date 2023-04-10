import cudf
import numpy as np
import cugraph
from dgl.sampling import node2vec_random_walk
import time
import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp


def load_ogb(name):
    data = DglNodePropPredDataset(name=name)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx

def load_livejournal():
    train_id = torch.load("/home/ubuntu/.dgl/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/.dgl/livejournal_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g=g.long()
    return g, None, None, None, splitted_idx

def time_randomwalk(graph,seeds,batchsize,walk_length,batchnum):
    """
    Test cost time of random walk
    """
    runs=5
    # start_vertices = cudf.Series(range(400), dtype=np.int32)
    time_list = []
    for j in range(0,batchnum):
        start = j * batchsize
        end = (j + 1) * batchsize
        sub_slicing = seeds[start:end]
        start_time = time.time()
        paths, weights, path_sizes = cugraph.random_walks(graph, random_walks_type='uniform', start_vertices=sub_slicing, max_depth=walk_length,legacy_result_type=True)
        end_time = time.time()
        # print("time:",end_time-start_time)
        time_list.append(end_time-start_time)

    print("Run {} seeds, {}times, mean run time: {:.6f} ms".format(len(seeds),len(time_list),np.sum(time_list)*1000))
    time_list = []
    for j in range(0,batchnum):
        start = j * batchsize
        end = (j + 1) * batchsize
        sub_slicing = seeds[start:end]
        start_time = time.time()
        paths, weights, path_sizes = cugraph.random_walks(graph, random_walks_type='uniform', start_vertices=sub_slicing, max_depth=walk_length,legacy_result_type=True)
        end_time = time.time()
        # print("time:",end_time-start_time)
        time_list.append(end_time-start_time)

    print("Run {} seeds, {}times, mean run time: {:.6f} ms".format(len(seeds),len(time_list),np.sum(time_list)*1000) )

dataset = load_ogb('ogbn-products')
#dataset = load_livejournal()
dgl_graph = dataset[0]
train_id = dataset[4]['train']
train_id = train_id.cpu().numpy()
train_id = cudf.Series(train_id)
dgl_graph = dgl_graph.to('cuda')
g_cugraph = dgl_graph.to_cugraph()
del dgl_graph
print("Timing random walks")
time_randomwalk(g_cugraph,train_id,65536,80,4)    
# coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
    # g = dgl.from_scipy(coo_matrix)
    # g = g.formats("csc")