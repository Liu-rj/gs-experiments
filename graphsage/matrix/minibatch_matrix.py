import gs
from gs.jit.passes import dce
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl import create_block
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from dgl.data import RedditDataset
import numpy as np
import time
import tqdm
import argparse
import os
from ctypes import *
from ctypes.util import *
import numpy as np


device = torch.device('cuda')
batch_time = 0


def load_custom_reddit():
    libgraphPath = '../libgraph.so'
    libgraph = CDLL(libgraphPath)
    libgraph.loadgraph.argtypes = [c_char_p]
    filename = "/home/ubuntu/NextDoorEval/NextDoor/input/reddit.data"
    if not os.path.exists(filename):
        raise Exception("'%s' do not exist" % (filename))

    graphPath = bytes(filename, encoding='utf8')
    libgraph.loadgraph(graphPath)
    libgraph.getEdgePairList.restype = np.ctypeslib.ndpointer(
        dtype=c_int, shape=(libgraph.numberOfEdges(), 2))

    print("Graph Loaded in C++")

    edges = libgraph.getEdgePairList()
    print("Number of Edges", libgraph.numberOfEdges())
    print("Number of Vertices", libgraph.numberOfVertices())
    src_ids = torch.tensor(edges[:, 0])
    dst_ids = torch.tensor(edges[:, 1])
    dgl_graph = dgl.graph((src_ids, dst_ids), idtype=torch.int64)
    num_nodes = dgl_graph.num_nodes()

    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(
        g.ndata["train_mask"][:num_nodes], as_tuple=True)[0]
    test_nid = torch.nonzero(
        g.ndata["test_mask"][:num_nodes], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"][:num_nodes], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "val": val_nid}
    feat = g.ndata['feat'][:num_nodes].clone()
    labels = g.ndata['label'][:num_nodes]
    g.ndata.clear()
    return dgl_graph, feat, labels, n_classes, splitted_idx


def graphsage_sampler(A: gs.Matrix, seeds: torch.Tensor, fanouts: list):
    global batch_time
    torch.cuda.synchronize()
    start = time.time()
    # torch.cuda.nvtx.range_push('graphsage')
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('slicing_sampling')
        # subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sampling')
        # subA = subA.columnwise_sampling(fanout, True)
        subA = gs.Matrix(
            A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, True))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('allindices')
        seeds = subA.all_indices()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('toblock')
        ret.insert(0, subA)
        # torch.cuda.nvtx.range_pop()
    input_nodes = seeds
    # torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    batch_time += time.time() - start
    return input_nodes, output_nodes, ret


def slicing_and_sampling_fuse(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if node.target == 'columnwise_sampling' and node.args[
                0].target == 'columnwise_slicing':
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    'fused_columnwise_slicing_sampling',
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ))
                node.replace_all_uses_with(new_node)
    gm.graph.lint()
    gm.recompile()
    return gm


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
    with torch.no_grad():
        for it, seeds in enumerate(seedloader):
            input_nodes, output_nodes, subMs = compiled_func(
                matrix, seeds, [25, 10])
            blocks = [block.to_dgl_block() for block in subMs]
            x = features[input_nodes]
            y = labels[output_nodes]
            ys.append(y)
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(graph, nid, model, batch_size, feat, label):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size,
                               feat)  # pred in buffer_device
        pred = pred[nid]
        label = label[nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = SAGE(features.shape[1], 256, n_classes, feat_device).to(device)
    # create sampler & dataloader
    m = gs.Matrix(gs.Graph(False))
    m.load_dgl_graph(g)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    # compiled_func = gs.jit.compile(
    #     func=graphsage_sampler, args=(m, torch.Tensor(), [25, 10]))
    # compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))
    compiled_func = graphsage_sampler
    train_seedloader = SeedGenerator(
        train_idx, batch_size=1024, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_idx, batch_size=1024, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    n_epoch = 10
    epoch_time = []
    epoch_sample = []

    for epoch in range(n_epoch):
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_loss = 0
        for it, seeds in enumerate(train_seedloader):
            input_nodes, output_nodes, subMs = compiled_func(m, seeds, [
                25, 10])
            blocks = [block.to_dgl_block() for block in subMs]
            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, m, compiled_func,
                       val_seedloader, features, labels)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        global batch_time
        epoch_sample.append(batch_time)
        batch_time = 0

        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | E2E Time {:.4f} s | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, total_loss / (it+1), acc.item(), epoch_time[-1], epoch_sample[-1], torch.cuda.max_memory_allocated() /
                      (1024 * 1024 * 1024)))

    print('Average epoch e2e time:', np.mean(epoch_time[3:]))
    print('Average epoch sample time:', np.mean(epoch_sample[3:]))

    print('Testing...')
    acc = layerwise_infer(g, test_idx, model,
                          batch_size=4096, feat=features, label=labels)
    print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode", default='cuda', choices=['cpu', 'cuda'],
                        help="Feature reside device. To cpu or gpu")
    args = parser.parse_args()
    print(args)
    feat_device = args.fmode
    # load and preprocess dataset
    print('Loading data')
    # g, features, labels, n_classes, splitted_idx = load_custom_reddit()
    g, features, labels, n_classes, splitted_idx = load_graph.load_reddit()
    g = g.to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    labels = labels.to(device)

    train(g, (features, labels, n_classes, train_idx,
              val_idx, test_mask), feat_device)
