import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
import dgl.function as fn
from dgl.utils import gather_pinned_tensor_rows
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import tqdm

class SAGEMeanAgg(nn.Module):
    def __init__(self, aggregator_type, dropout):
        super(SAGEMeanAgg, self).__init__()
        assert aggregator_type == 'mean'
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat_tuple):
        with graph.local_scope():
            feat_src, h_self = feat_tuple
            feat_src = self.dropout(feat_src)
            h_self = self.dropout(h_self)
            msg_fn = fn.copy_src('h', 'm')
            graph.srcdata['h'] = feat_src
            graph.update_all(msg_fn, fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h = torch.add(h_neigh, h_self)
            h = F.relu(h)
            return h

class GraphSAGE_DGL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, use_uva):
        super().__init__()
        self.num_layers=num_layers
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, 'mean'))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.use_uva = use_uva

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

