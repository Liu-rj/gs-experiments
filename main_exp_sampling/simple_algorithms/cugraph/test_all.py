import cugraph
import cudf
import time
import numpy as np
import torch
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import dgl
from tqdm import tqdm


def load_ogb(name):
    data = DglNodePropPredDataset(name=name)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    feat = g.ndata["feat"]
    labels = labels[:, 0]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


def load_livejournal():
    train_id = torch.load("/home/ubuntu/data/livejournal/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx["train"] = train_id
    coo_matrix = sp.load_npz("/home/ubuntu/.dgl/livejournal_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.long()
    return g, None, None, None, splitted_idx


def time_graphsage(graph, seeds, batchsize, fanout, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            result, duration = cugraph.uniform_neighbor_sample(
                graph, sub_slicing, fanout, with_replacement=False
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


def time_deepwalk(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes, duration = cugraph.random_walks(
                graph,
                random_walks_type="uniform",
                start_vertices=sub_slicing,
                max_depth=walk_length,
                legacy_result_type=True,
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


def time_node2vec(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes, duration = cugraph.node2vec(
                graph,
                start_vertices=sub_slicing,
                max_depth=walk_length,
                compress_result=True,
                p=2.0,
                q=0.5,
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="livejournal",
    choices=["ogbn-products", "livejournal"],
    help="which dataset to load for training",
)
args = parser.parse_args()

if args.dataset == "livejournal":
    dataset = load_livejournal()
elif args.dataset == "ogbn-products":
    dataset = load_ogb("ogbn-products")
else:
    raise NotImplementedError

dgl_graph = dataset[0]
train_id = dataset[4]["train"]
train_id = train_id.cpu().numpy()
index = np.random.permutation(train_id.shape[0])
permuted_nid = cudf.Series(train_id[index])
dgl_graph = dgl_graph.to("cuda")
G = dgl_graph.to_cugraph()
del dgl_graph

print("Timing graphsage", flush=True)
batchsize = 512
fanout = [25, 10]
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
print(batchnum)
time_graphsage(G, permuted_nid, batchsize, fanout, batchnum)


print("Timing deepmwalk", flush=True)
batchsize = 1024
walk_len = 80
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
time_deepwalk(G, permuted_nid, batchsize, walk_len, batchnum)


print("Timing node2vec", flush=True)
batchsize = 1024
walk_len = 80
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
time_node2vec(G, permuted_nid, batchsize, walk_len, batchnum)
