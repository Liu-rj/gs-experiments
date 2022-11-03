import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import tqdm
import dgl.nn as dglnn


def degree_ndata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        # g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g.ndata['v'] = torch.ones(g.number_of_nodes(), device=g.device)
        return g.ndata['v']


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            return g.dstdata['y']


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            #g.edata['w'] = w
            #g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = torch.cat([g.dstdata['x'], g.dstdata['y']], 1)
            return self.W(h)


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, conv=GraphConv, dropout=0):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(conv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(conv(n_hidden, n_hidden))
        self.convs.append(conv(n_hidden, n_classes))

    def forward(self, blocks, x):
        if not isinstance(blocks, list):
            blocks = [blocks] * len(self.convs)
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = self.dropout(x)
            x = conv(block, x, block.edata['w'])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

    def inference(self, g, x, w, batch_size, device, num_workers):
        with torch.no_grad():
            for l, layer in enumerate(self.convs):
                y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(
                    self.convs) - 1 else self.n_classes)

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    torch.arange(g.number_of_nodes()).cuda(),
                    sampler,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=num_workers, device=device)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0]

                    block = block.int().to(device)
                    h = x[input_nodes].to(device)
                    w_block = w[block.edata[dgl.EID]].to(device)
                    h = layer(block, h, w_block)
                    if l != len(self.convs) - 1:
                        h = F.relu(h)

                    y[output_nodes] = h.cpu()

                x = y
            return y


class SAGEModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, feat_device):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.feat_device = feat_device

    def forward(self, blocks, x):
        h = x.to('cuda') if self.feat_device == 'cpu' else x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, feat):
        """Conduct layer-wise inference to get all the node embeddings."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
