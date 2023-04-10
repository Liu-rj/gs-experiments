# from cugraph.experimental.datasets import karate
# import cudf
# import numpy as np
# import cugraph
# import cudf

# # # 创建一个COO格式的边缘列表
# # coo_data = [
# #     (0, 1),
# #     (1, 2),
# #     (2, 3),
# #     (3, 0)
# # ]

# # # 将数据转换为cuDF DataFrame
# # edges_df = cudf.DataFrame(coo_data, columns=["src", "dst"])

# # # 创建cugraph.Graph对象
# # G = cugraph.Graph()

# # # 向图中添加边缘数据
# # G.from_cudf_edgelist(edges_df, source="src", destination="dst")

# # # 打印图的基本信息
# # print("node num:", G.number_of_nodes())
# # print("edge num:", G.number_of_edges())


# import time

# from dgl.sampling import node2vec_random_walk
# import time
# import dgl
# import torch
# from dgl.data import RedditDataset
# from ogb.nodeproppred import DglNodePropPredDataset


# def load_reddit():
#     data = RedditDataset(self_loop=True)
#     g = data[0]
#     n_classes = data.num_classes
#     train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
#     test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
#     val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
#     splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
#     feat = g.ndata['feat']
#     labels = g.ndata['label']
#     g.ndata.clear()
#     return g, feat, labels, n_classes, splitted_idx


# def load_ogb(name):
#     data = DglNodePropPredDataset(name=name)
#     splitted_idx = data.get_idx_split()
#     g, labels = data[0]
#     feat = g.ndata['feat']
#     labels = labels[:, 0]
#     n_classes = len(
#         torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
#     g.ndata.clear()
#     g = dgl.remove_self_loop(g)
#     g = dgl.add_self_loop(g)
#     return g, feat, labels, n_classes, splitted_idx
# def load_livejournal():
#     bin_path = "/home/ubuntu/data/livejournal/livejournal.bin"
#     g_list, _ = dgl.load_graphs(bin_path)
#     g = g_list[0]
#     print("graph loaded")
#     train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
#     test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
#     val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
#     g.ndata.clear()
#     # features = np.random.rand(g.num_nodes(), 128)
#     # labels = np.random.randint(0, 3, size=g.num_nodes())
#     # feat = torch.tensor(features, dtype=torch.float32)
#     # labels = torch.tensor(labels, dtype=torch.int64)
#     # n_classes = 3
#     # coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
#     # g = dgl.from_scipy(coo_matrix)
#     # g = g.formats("csc")
#     g=g.long()
#     return g, None,None,None,None
# def time_randomwalk(graph):
#     """
#     Test cost time of random walk
#     """
#     runs=3
#     start_vertices = cudf.Series(range(40000), dtype=np.int32)
#     time_list = []


#     for i in range(runs):
#         start_time = time.time()
#         paths, weights, path_sizes = cugraph.node2vec(graph, start_vertices, 100,
#                                               True, 2.0, 0.5)
#         time_list.append(time.time()-start_time)
#     end_time = time.time()
#     cost_time_avg = (end_time - start_time) / runs
#     print("Run {} trials, mean run time: {:.3f}s".format(runs, np.mean(time_list[1:])) )

# dataset = load_livejournal()
# dgl_graph = dataset[0]
# dgl_graph = dgl_graph.to('cuda')
# g_cugraph = dgl_graph.to_cugraph()
# del dgl_graph
# print("Timing random walks")
# time_randomwalk(g_cugraph)


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
        paths, weights, path_sizes = cugraph.node2vec(graph, start_vertices=sub_slicing,max_depth=100,compress_result=True, p=2.0, q=0.5)
        end_time = time.time()
        # print("time:",end_time-start_time)
        time_list.append(end_time-start_time)
    print("Run {} seeds, {}times, mean run time: {:.6f} ms".format(len(seeds),len(time_list),np.sum(time_list)*1000) )

#dataset = load_ogb('ogbn-products')
dataset = load_livejournal()
dgl_graph = dataset[0]
train_id = dataset[4]['train']
train_id = train_id.cpu().numpy()
train_id = cudf.Series(train_id)
dgl_graph = dgl_graph.to('cuda')
g_cugraph = dgl_graph.to_cugraph()
del dgl_graph
print("Timing random walks")
time_randomwalk(g_cugraph,train_id,128,80,9432)    
# coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
    # g = dgl.from_scipy(coo_matrix)
    # g = g.formats("csc")