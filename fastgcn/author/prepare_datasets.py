import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import scipy.sparse as sp


def transfer_reddit(output_path):
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.clear()
    sp.save_npz(output_path + "/reddit_adj.npz", g.adj(scipy_fmt='coo'))


def transfer_ogb(name, root, output_path):
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    train_ids, val_ids, test_ids = splitted_idx['train'].numpy(
    ), splitted_idx['valid'].numpy(), splitted_idx['test'].numpy()
    g, labels = data[0]
    feats = g.ndata['feat'].numpy()
    # feats[:, 0] = np.log(feats[:, 0] + 1.0)
    # feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    labels = labels.numpy()
    train_labels = labels[train_ids]
    test_labels = labels[test_ids]
    val_labels = labels[val_ids]
    g.ndata.clear()
    g = g.long()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    sp.save_npz(output_path + f"/{name}_adj.npz", g.adj(scipy_fmt='coo'))
    np.savez(output_path + f"/{name}.npz", feats=feats, y_train=train_labels, y_val=val_labels, y_test=test_labels,
             train_index=train_ids,
             val_index=val_ids, test_index=test_ids, n_classes=n_classes)


def transfer_friendster(output_path):
    bin_path = "../../datasets/friendster/friendster_coo_with_feature_large.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    g = g_list[0]
    print("graph loaded")
    train_ids = torch.nonzero(
        g.ndata["train_mask"].long(), as_tuple=True)[0].numpy()
    test_ids = torch.nonzero(
        g.ndata["test_mask"].long(), as_tuple=True)[0].numpy()
    val_ids = torch.nonzero(
        g.ndata["val_mask"].long(), as_tuple=True)[0].numpy()
    g = g.long()
    feats = np.random.rand(g.num_nodes(), 64)
    labels = np.random.randint(0, 2, size=g.num_nodes())
    train_labels = labels[train_ids]
    test_labels = labels[test_ids]
    val_labels = labels[val_ids]
    n_classes = 2
    g.ndata.clear()
    g.edata.clear()
    print("adding self loop...")
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    sp.save_npz(output_path + "/friendster_adj.npz", g.adj(scipy_fmt='coo'))
    np.savez(output_path + f"/friendster.npz", feats=feats, y_train=train_labels, y_val=val_labels, y_test=test_labels,
             train_index=train_ids,
             val_index=val_ids, test_index=test_ids, n_classes=n_classes)


if __name__ == "__main__":
    # transfer_reddit("data/reddit_dgl")
    # print('start transfer products')
    # transfer_ogb('ogbn-products', '../../datasets', "data/products")
    # print('start transfer papers100m')
    # transfer_ogb('ogbn-papers100M', '../../datasets', "data/papers")
    print('start transfer friendster')
    transfer_friendster("data/friendster")
