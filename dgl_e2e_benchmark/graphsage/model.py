from sampler import *
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
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, 'mean'))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.use_uva = use_uva

    def forward(self, blocks, h):
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
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
