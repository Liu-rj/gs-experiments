import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import pin_memory_inplace
from load_graph import load_custom_reddit
from model import *
from graphsage_sampler import *
import KHopSamplingPy3 as NextDoorKHopSampler
import ctypes
from numpy.ctypeslib import ndpointer
import time
import argparse


device = torch.device('cuda')
file_path = "/home/ubuntu/NextDoorEval/NextDoor/input/reddit.data"


def evaluate_dgl(model, graph, dataloader, features, labels):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = features[input_nodes]
            y = labels[output_nodes]
            ys.append(y)
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


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
        pred = model.inference(graph, device, batch_size, feat) # pred in buffer_device
        pred = pred[nid]
        label = label[nid].to(pred.device)
        return MF.accuracy(pred, label)


def train_dgl(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = GraphSAGE_DGL(features.shape[1], 256, n_classes, feat_device).to(device)
    # create sampler & dataloader
    sampler = NeighborSampler([25, 10])
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    accumulated_time = 0
    
    for epoch in range(10):
        torch.cuda.synchronize()
        start = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate_dgl(model, g, val_dataloader, features, labels)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
        torch.cuda.synchronize()
        accumulated_time += time.time() - start
        print(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 'GB')
    
    print('Average epoch time:', accumulated_time / 10)

    print('Testing...')
    acc = layerwise_infer(g, test_idx, model, batch_size=4096, feat=features, label=labels)
    print("Test Accuracy {:.4f}".format(acc.item()))


def train_nextdoor(g, dataset, feat_device, use_uva):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = GraphSAGE_Nextdoor(features.shape[1], 256, n_classes, 2, feat_device, use_uva).to(device)
    # create sampler & dataloader
    NextDoorKHopSampler.initSampling(file_path)
    lib = ctypes.CDLL("./KHopSamplingPy3.so")
    print("NextDoorKHopSampler.finalSampleLength() ", NextDoorKHopSampler.finalSampleLength())
    lib.finalSamplesArray.restype = ndpointer(dtype=ctypes.c_int, shape=(min(NextDoorKHopSampler.finalSampleLength(), 2**28)))
    sampler = NextdoorKhopSampler([25, 10], sampler=NextDoorKHopSampler, lib=lib)
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    accumulated_time = 0
    
    for epoch in range(10):
        torch.cuda.synchronize()
        start = time.time()
        # sampler.sample_time = 0
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
        accumulated_time += time.time() - start
        print(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 'GB')

    print('Average epoch time:', accumulated_time / 10)
    
    print('Testing...')
    acc = layerwise_infer(g, test_idx, model, batch_size=4096, feat=features, label=labels)
    print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode", default='cuda', choices=['cpu', 'cuda', 'uva'],
                        help="Feature reside device. To cpu or gpu")
    args = parser.parse_args()
    feat_device = 'cuda'
    use_uva = False
    if args.fmode != 'cuda':
        feat_device = 'cpu'
        if args.fmode == 'uva':
            use_uva = True
    # load and preprocess dataset
    print('Loading data')
    g, features, labels, n_classes, splitted_idx = load_custom_reddit(file_path)
    g = g.to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    if use_uva:
        feat_ndarray = pin_memory_inplace(features)
    labels = labels.to(device)

    # train_dgl(g, (features, labels, n_classes, train_idx, val_idx, test_mask), feat_device)
    train_nextdoor(g, (features, labels, n_classes, train_idx, val_idx, test_mask), feat_device, use_uva)