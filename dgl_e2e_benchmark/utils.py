import inspect
import json
import os
import pickle
import shutil
import time
import zipfile
from functools import partial, reduce, wraps
from timeit import default_timer
import dgl.backend as F
import numpy as np
import pandas
import requests
import torch
from ogb.nodeproppred import DglNodePropPredDataset

import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np

from dgl.data.dgl_dataset import DGLDataset, DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
from dgl.convert import from_scipy
from dgl.transforms import reorder_graph
import dgl


def load_friendster():
    bin_path = "/mzydata/data/friendster_coo_with_feature_large.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    g = g_list[0]
    print("graph loaded")
    train_nid = torch.nonzero(g.ndata["train_mask"].long(), as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"].long(), as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"].long(), as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    g=g.long()
    features = np.random.rand(g.num_nodes(), 64)
    labels = np.random.randint(0, 2, size=g.num_nodes())
    feat = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    n_classes = 2
    g.ndata.clear()
    g.edata.clear()
    print("adding self loop...")
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    splitted_idx = dict()
    splitted_idx['train'] = train_nid
    splitted_idx['valid']=val_nid
    splitted_idx['test']=test_nid
    print(g)
    return g, feat, labels, n_classes, splitted_idx

def get_rand_type():
    val = np.random.uniform()
    if val < 0.1:
      return 0
    elif val < 0.4:
      return 1
    return 2

def get_graph(format=None):
    g = None
    # if os.path.exists(bin_path):
    # g_list, _ = dgl.load_graphs(bin_path)
    # g = g_list[0]
   # else:
    g = get_friendster()
    g = dgl.to_bidirected(g)
    g = dgl.to_simple(g)
    g = dgl.compact_graphs(g)
    num_nodes = g.num_nodes()
    node_types = []
    print("generating mask begin")
    for i in range(0, num_nodes):
        node_types.append(get_rand_type())
    node_types = np.array(node_types)
    g.ndata['node_type'] = F.tensor(node_types, dtype=F.data_type_dict['int32'])
    # features = np.random.rand(num_nodes, 128)
    # labels = np.random.randint(0, 3, size=num_nodes)
    train_mask = (node_types == 0)
    val_mask = (node_types == 1)
    test_mask = (node_types == 2)
    print("generating mask done")
    print("generating mask train mask tensor")
    g.ndata['train_mask'] = generate_mask_tensor(train_mask)
    print("generating mask train val tensor")
    g.ndata['val_mask'] = generate_mask_tensor(val_mask)
    print("generating mask train test tensor")
    g.ndata['test_mask'] = generate_mask_tensor(test_mask)
    g.ndata.pop('node_type')
    print("generating mask train feat tensor")
    # g.ndata['feat'] = F.tensor(features, dtype=F.data_type_dict['float32'])
    print("generating mask train label tensor")
    # g.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
    g.ndata.pop('_ID')
    new_name = "/mzydata/data/friendster_coo_with_feature_large.bin"
    #g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    print("saving graph...")
    dgl.save_graphs(new_name, [g])
    return g








def get_friendster():
    # df = pandas.read_csv("/home/ubuntu/data/com-friendster.ungraph.txt",sep="\t",skiprows=1,header=None,names=["src", "dst"])
    df = pandas.read_csv("/mzydata/data/com-friendster.ungraph.txt",sep="\t",skiprows=4,header=None,names=["src", "dst"])
   
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    return dgl.graph((src, dst))

if __name__ == '__main__':
    graph = get_graph()
    print(graph.nodes())
    print(graph)

