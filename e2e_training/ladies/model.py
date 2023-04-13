import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.function as fn


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {
                      'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']
    

class GCNModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(n_hidden, n_hidden))
        self.convs.append(GraphConv(n_hidden, n_classes))

    def forward(self, blocks, x):
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = conv(block, x, edge_weight=block.edata['w'])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x
