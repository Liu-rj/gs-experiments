import gs
from gs.jit.passes import dce
from gs.utils import SeedGenerator
from gs.utils import load_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import numpy as np
import time
import tqdm
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset


device = torch.device('cuda')
time_list = []
batch_time = 0


def load_ogbn_papers100M():
    data = DglNodePropPredDataset(
        name="ogbn-papers100M", root="../../datasets")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    print(g)
    feat = g.ndata['feat']
    labels = labels[:, 0].long()
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


def graphsage_sampler(A: gs.Matrix, seeds: torch.Tensor, fanouts: list):
    global batch_time

    torch.cuda.nvtx.range_push('graphsage')
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        # torch.cuda.synchronize()
        # start = time.time()
        torch.cuda.nvtx.range_push('slicing_sampling')
        # subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sampling')
        # subA = subA.columnwise_sampling(fanout, True)
        subA = gs.Matrix(
            A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, True))
        torch.cuda.nvtx.range_pop()
        # torch.cuda.synchronize()
        # batch_time += time.time() - start
        torch.cuda.nvtx.range_push('allindices')
        seeds = subA.all_indices()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('toblock')
        ret.insert(0, subA.to_dgl_block())
        torch.cuda.nvtx.range_pop()
    input_nodes = seeds
    torch.cuda.nvtx.range_pop()

    return input_nodes, output_nodes, ret


class SAGE(nn.Module):
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


def evaluate(model, matrix, compiled_func, seedloader, features, labels):
    model.eval()
    ys = []
    y_hats = []
    for it, seeds in enumerate(seedloader):
        input_nodes, output_nodes, blocks = compiled_func(
            matrix, seeds, [25, 10])
        with torch.no_grad():
            x = features[input_nodes].to(device)
            y = labels[output_nodes].to(device)
            ys.append(y)
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(graph, nid, model, batch_size, feat, label):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size,
                               feat)  # pred in buffer_device
        label = label[nid].to(pred.device)
        pred = pred[nid]
        is_labeled = label == label
        return MF.accuracy(pred[is_labeled], label[is_labeled])


def train(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = SAGE(features.shape[1], 256, n_classes, feat_device).to(device)
    # create sampler & dataloader
    m = gs.Matrix(gs.Graph(False))
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    pinned_csc_indptr = csc_indptr.pin_memory()
    pinned_csc_indices = csc_indices.pin_memory()
    m._graph._CAPI_load_csc(pinned_csc_indptr, pinned_csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    compiled_func = graphsage_sampler
    # compiled_func = gs.jit.compile(
    #     func=graphsage_sampler, args=(m, torch.Tensor(), [25, 10]))
    # compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))
    train_seedloader = SeedGenerator(
        train_idx, batch_size=1024, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_idx, batch_size=1024, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    n_epoch = 3

    for epoch in range(n_epoch):
        # torch.cuda.synchronize()
        # start = time.time()

        model.train()
        total_loss = 0
        for it, seeds in enumerate(train_seedloader):
            seeds = seeds.to(device)
            input_nodes, output_nodes, blocks = compiled_func(m, seeds, [
                                                              25, 10])
            x = features[input_nodes].to(device)
            y = labels[output_nodes].to(device)
            y_hat = model(blocks, x)
            is_labeled = y == y
            loss = F.cross_entropy(y_hat[is_labeled], y[is_labeled])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, m, compiled_func,
                       val_seedloader, features, labels)

        # torch.cuda.synchronize()
        # end = time.time()
        # time_list.append(end - start)
        global batch_time
        time_list.append(batch_time)
        batch_time = 0

        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | E2E Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, total_loss / (it+1), acc.item(), time_list[-1], torch.cuda.max_memory_allocated() /
                      (1024 * 1024 * 1024)))

    print('Average epoch time:', np.mean(time_list[3:]), 'seconds')

    # print('Testing...')
    # acc = layerwise_infer(g, test_idx, model,
    #                       batch_size=1024, feat=features, label=labels)
    # print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    feat_device = 'cpu'
    # load and preprocess dataset
    print('Loading data')
    g, features, labels, n_classes, splitted_idx = load_ogbn_papers100M()
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']

    train(g, (features, labels, n_classes, train_mask,
              val_mask, test_mask), feat_device)
