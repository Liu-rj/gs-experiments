import dgl.backend as F
import numpy as np
import pandas
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
from dgl.data import RedditDataset


def transfer_reddit(output_path):
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0].numpy()
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0].numpy()
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0].numpy()
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata['feat'].numpy()
    labels = g.ndata['label'].numpy()
    g.ndata.clear()
    sp.save_npz(output_path + "/reddit_adj.npz", g.adj(scipy_fmt='coo'))
    np.savez(output_path + f"/reddit.npz", feats=feat, train_index=train_nid,
             val_index=val_nid, test_index=test_nid, n_classes=n_classes)


def transfer_ogb(name, root, output_path):
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    train_ids, val_ids, test_ids = splitted_idx['train'].numpy(
    ), splitted_idx['valid'].numpy(), splitted_idx['test'].numpy()
    g, labels = data[0]
    feats = g.ndata['feat'].numpy()
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    labels = labels.numpy()
    g.ndata.clear()
    g = g.long()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    sp.save_npz(output_path + f"/{name}_adj.npz", g.adj(scipy_fmt='coo'))
    np.savez(output_path + f"/{name}.npz", feats=feats, train_index=train_ids,
             val_index=val_ids, test_index=test_ids, n_classes=n_classes)


def get_rand_type():
    val = np.random.uniform()
    if val < 0.1:
        return 0
    elif val < 0.4:
        return 1
    return 2


def get_friendster(format=None):
    df = pandas.read_csv("/home/ubuntu/dataset/friendster/com-friendster.ungraph.txt",
                         sep="\t", skiprows=4, header=None, names=["src", "dst"])
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g = dgl.to_simple(g)
    g = dgl.compact_graphs(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_nodes = g.num_nodes()
    node_types = []
    print("generating mask begin")
    for i in range(0, num_nodes):
        node_types.append(get_rand_type())
    node_types = np.array(node_types)
    g.ndata['node_type'] = F.tensor(
        node_types, dtype=F.data_type_dict['int32'])
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
    new_name = "/home/ubuntu/dataset/friendster/friendster.bin"
    # g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    print("saving graph...")
    dgl.save_graphs(new_name, [g])

    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0].numpy()
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0].numpy()
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0].numpy()
    sp.save_npz("/home/ubuntu/dataset/friendster/friendster_adj.npz",
                g.adj(scipy_fmt='coo'))
    np.savez("/home/ubuntu/dataset/friendster/friendster.npz",
             train_index=train_nid, val_index=val_nid, test_index=test_nid)
    return g


def get_livejournal(format=None):
    df = pandas.read_csv("/home/ubuntu/dataset/livejournal/soc-LiveJournal1.txt",
                         sep="\t", skiprows=4, header=None, names=["src", "dst"])
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    g = dgl.graph((src, dst))
    g = dgl.to_simple(g)
    g = dgl.compact_graphs(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    num_nodes = g.num_nodes()
    node_types = []
    print("generating mask begin")
    for i in range(0, num_nodes):
        node_types.append(get_rand_type())
    node_types = np.array(node_types)
    g.ndata['node_type'] = F.tensor(
        node_types, dtype=F.data_type_dict['int32'])
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
    new_name = "/home/ubuntu/dataset/livejournal/livejournal.bin"
    # g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    print("saving graph...")
    dgl.save_graphs(new_name, [g])

    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0].numpy()
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0].numpy()
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0].numpy()
    sp.save_npz("/home/ubuntu/dataset/livejournal/livejournal_adj.npz",
                g.adj(scipy_fmt='coo'))
    np.savez("/home/ubuntu/dataset/livejournal/livejournal.npz",
             train_index=train_nid, val_index=val_nid, test_index=test_nid)
    return g


if __name__ == '__main__':
    graph = get_livejournal()
    print(graph.nodes())
    print(graph)

    graph = get_friendster()
    print(graph.nodes())
    print(graph)
    print('transfer txt graph successful')
