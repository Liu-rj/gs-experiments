import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            return g.dstdata['y']


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = torch.cat([g.dstdata['x'], g.dstdata['y']], 1)
            return self.W(h)


class Model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size))
        for i in range(n_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size))
        self.layers.append(SAGEConv(hid_size, out_size))
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h
