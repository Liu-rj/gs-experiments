import dgl
import torch
from dgl.data import RedditDataset
from dgl.data.utils import generate_mask_tensor
from dgl.convert import from_scipy
from dgl.transforms import reorder_graph
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os
from ctypes import *
from ctypes.util import *
import networkx as nx
import random
import numpy as np


def load_reddit():
    coo_adj = sp.load_npz('../author/data/reddit/reddit_adj.npz')
    graph = from_scipy(coo_adj)
    # features and labels
    reddit_data = np.load('../author/data/reddit/reddit.npz')
    features = reddit_data["feats"]
    y_train, y_val, y_test = reddit_data['y_train'], reddit_data['y_val'], reddit_data['y_test']
    # tarin/val/test indices
    train_index, val_index, test_index = reddit_data[
        'train_index'], reddit_data['val_index'], reddit_data['test_index']
    graph = reorder_graph(
        graph, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    n_classes = 41
    train_nid = torch.tensor(train_index, dtype=torch.int64)
    val_nid = torch.tensor(val_index, dtype=torch.int64)
    test_nid = torch.tensor(test_index, dtype=torch.int64)
    splitted_idx = {"train": train_nid, "test": test_nid, "val": val_nid}
    feat = torch.tensor(features, dtype=torch.float32)
    labels = np.zeros([graph.num_nodes()], dtype=int)
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    labels = torch.tensor(labels, dtype=torch.int64)
    return graph, feat, labels, n_classes, splitted_idx


def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products", root="../datasets")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx
