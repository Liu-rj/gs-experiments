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


class GraphSAGE_Nextdoor(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 feature_device,
                 use_uva):
        super(GraphSAGE_Nextdoor, self).__init__()
        if feature_device == 'cuda':
            assert use_uva == False
        self.hid_size, self.out_size = n_hidden, n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.aggregator = SAGEMeanAgg('mean', 0.5)
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(in_feats, n_hidden))
        self.linear_layers.append(nn.Linear(n_hidden, n_classes))
        self.n_layers = n_layers
        self.dropout = nn.Dropout(0.5)
        self.feature_device = feature_device
        self.use_uva = use_uva

    def forward(self, blocks, features, samples):
        blocks.reverse()
        if self.feature_device == 'cuda':
            hidden = [features[sample] for sample in samples]
        elif self.use_uva:
            hidden = [gather_pinned_tensor_rows(features, sample) for sample in samples]
        elif self.feature_device == 'cpu':
            hidden = [features[sample].to('cuda') for sample in samples]
        for layer in range(self.n_layers):
            next_hidden = []
            for hop in range(self.n_layers - layer):
                h = self.aggregator(blocks[hop], (hidden[hop + 1], hidden[hop]))
                next_hidden.append(self.linear_layers[layer](h))
            hidden = next_hidden
        h = hidden[0]
        return h
    
    def inference(self, g, device, batch_size, feat):
        """Conduct layer-wise inference to get all the node embeddings."""
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
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
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y


class GraphSAGE_DGL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, feat_device):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(SAGEConv(hid_size, out_size, 'mean'))
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
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
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
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y