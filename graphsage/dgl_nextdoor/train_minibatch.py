import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import pin_memory_inplace
from dgl.data import RedditDataset
from model import *
from sampler import *
import time
import argparse
import os
from ctypes import *
from ctypes.util import *
import numpy as np
from gs.utils import load_graph
from tqdm import tqdm


device = torch.device('cuda')


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


def evaluate_nextdoor(model, graph, dataloader, features, labels):
    model.eval()
    ys = []
    y_hats = []
    for it, (samples, blocks) in enumerate(dataloader):
        with torch.no_grad():
            y = labels[samples[0]]
            ys.append(y)
            y_hats.append(model(blocks, features, samples))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(graph, nid, model, batch_size, feat, label):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size,
                               feat)  # pred in buffer_device
        pred = pred[nid]
        label = label[nid].to(pred.device)
        return MF.accuracy(pred, label)


def train_dgl(g, dataset, feat_device):
    batch_size = 64
    num_layers = 3
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = GraphSAGE_DGL(
        features.shape[1], 64, n_classes, num_layers, feat_device).to(device)
    # create sampler & dataloader
    sampler = DGLNeighborSampler([25, 10, 10])
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=batch_size, shuffle=True,
                                  drop_last=False, num_workers=0)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=batch_size, shuffle=True,
                                drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    time_list = []
    epoch_sample = []
    n_epoch = 2

    for epoch in range(n_epoch):
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                x = features[input_nodes]
                y = labels[output_nodes]
                ys.append(y)
                y_hats.append(model(blocks, x))
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))

        torch.cuda.synchronize()
        time_list.append(time.time() - start)
        epoch_sample.append(sampler.sample_time)
        sampler.sample_time = 0

        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | E2E Time {:.4f} s | Epoch Sampling Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, total_loss / (it+1), acc.item(), time_list[-1], epoch_sample[-1], torch.cuda.max_memory_allocated() /
                      (1024 * 1024 * 1024)))

    print('Average epoch end2end time:', np.mean(time_list[3:]))
    print('Average epoch sampling time:', np.mean(epoch_sample[3:]))


def train_nextdoor(g, dataset, feat_device, use_uva):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = GraphSAGE_Nextdoor(
        features.shape[1], 256, n_classes, 2, feat_device, use_uva).to(device)
    # create sampler & dataloader
    sampler = NextdoorKhopSampler([25, 10], file_path=file_path)
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    accumulated_time = 0
    n_epoch = 10

    for epoch in range(n_epoch):
        torch.cuda.synchronize()
        start = time.time()
        model.train()
        total_loss = 0
        for it, (samples, blocks) in enumerate(train_dataloader):
            y = labels[samples[0]]
            y_hat = model(blocks, features, samples)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate_nextdoor(model, g, val_dataloader, features, labels)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
        torch.cuda.synchronize()
        # accumulated_time += sampler.sample_time
        # sampler.sample_time = 0
        accumulated_time += time.time() - start
        # print(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 'GB')

    print('Average epoch time:', accumulated_time / n_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode", default='cuda', choices=['cpu', 'cuda', 'uva'],
                        help="Feature reside device. To cpu or gpu")
    args = parser.parse_args()
    print(args)
    feat_device = 'cuda'
    use_uva = False
    if args.fmode != 'cuda':
        feat_device = 'cpu'
        if args.fmode == 'uva':
            use_uva = True
    # load and preprocess dataset
    print('Loading data')
    # g, features, labels, n_classes, splitted_idx = load_custom_reddit()
    g, features, labels, n_classes, splitted_idx = load_graph.load_reddit()
    print(g)
    g = g.to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    if use_uva:
        feat_ndarray = pin_memory_inplace(features)
    labels = labels.to(device)

    train_dgl(g, (features, labels, n_classes, train_idx,
              val_idx, test_mask), feat_device)
    # train_nextdoor(g, (features, labels, n_classes, train_idx,
    #                val_idx, test_mask), feat_device, use_uva)
