import argparse
import time
import numpy as np
import torch
import load_graph
from sampler import *
from model import *
from dgl.utils import pin_memory_inplace


torch.ops.load_library("./nextdoor/build/libnextdoor.so")


def evaluate_dgl(model, graph, features, labels, nid, seeds, fanout):
    model.eval()
    with torch.no_grad():
        blocks = neighborsampler_dgl(graph, seeds, fanout)
        logits = model(blocks, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def evaluate_nextdoor(model, features, labels, nid, sampler, seeds, fanout):
    model.eval()
    with torch.no_grad():
        blocks, samples = neighborsampler_nextdoor(sampler, seeds, fanout)
        logits = model(blocks, features, samples)
        labels = labels[seeds]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_dgl(args, file_name):
    # load and preprocess dataset
    g, features, labels, n_classes, splitted_idx = load_graph.load_custom_reddit(
        file_name)
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    torch.cuda.set_device(0)
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    g = g.to('cuda')

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # create GraphSAGE model
    model = GraphSAGE_DGL(features.shape[1],
                          args.n_hidden,
                          n_classes,
                          'cuda')

    model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    seeds = g.nodes()
    fanout = [25, 10]
    for epoch in range(args.n_epochs):
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        # forward
        blocks = neighborsampler_dgl(g, seeds, fanout)
        logits = model(blocks, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        dur.append(time.time() - t0)

        acc = evaluate_dgl(model, g, features, labels, val_nid, seeds, fanout)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print('Average Training time:', sum(dur) / len(dur))
    acc = evaluate_dgl(model, g, features, labels, test_nid, seeds, fanout)
    print("Test Accuracy {:.4f}".format(acc))


def train_nextdoor(args, file_name, feat_device, use_uva):
    # load and preprocess dataset
    g, features, labels, n_classes, splitted_idx = load_graph.load_custom_reddit(
        file_name)
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    torch.cuda.set_device(0)
    features = features.to(feat_device)
    if use_uva:
        feat_ndarray = pin_memory_inplace(features)
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    g = g.to('cuda')

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # create GraphSAGE model
    model = GraphSAGE_Nextdoor(features.shape[1],
                               args.n_hidden,
                               n_classes,
                               args.n_layers,
                               feat_device,
                               use_uva)

    model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    seeds = g.nodes()
    fanout = [25, 10]

    sampler = torch.classes.my_classes.NextdoorKHopSampler(file_name)
    sampler.initSampling()

    accumulate_time = 0

    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        torch.cuda.synchronize()
        start = time.time()
        samples, blocks = neighborsampler_nextdoor(sampler, seeds, fanout)
        torch.cuda.synchronize()
        end = time.time()
        logits = model(blocks, features, samples)
        loss = F.cross_entropy(logits, labels[seeds])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate_nextdoor(model, features, labels,
                                val_nid, sampler, seeds, fanout)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))
        accumulate_time += end - start

    print('Average epoch time:', accumulate_time / args.n_epochs)

    print()
    acc = evaluate_nextdoor(model, features, labels,
                            test_nid, sampler, seeds, fanout)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    data_path = "/home/ubuntu/NextDoorEval/NextDoor/input/reddit.data"
    train_dgl(args, data_path)
    # train_nextdoor(args, data_path, 'cuda', use_uva=False)
