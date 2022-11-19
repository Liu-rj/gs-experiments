
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import pin_memory_inplace
from model import *
from sampler import *
import time
import argparse
import os
from ctypes import *
from ctypes.util import *
import numpy as np
from load_graph import *
import torch.nn.functional as F


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train_dgl(dataset, config):
    feat_device = config['feat_device']
    device = config['device']
    use_uva = config['use_uva']
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']

    if use_uva:
        features, labels = features.pin_memory(), labels.pin_memory()
    elif config['device'] == 'cuda':
        g, train_nid, val_nid = g.to('cuda'), train_nid.to(
            'cuda'), val_nid.to('cuda')
        features = features.to('cuda')
        labels = labels.to('cuda')
    else:
        features = features.to('cuda')
        labels = labels.to('cuda')
    batch_size = config['batch_size']
    num_layers = 3
    model = GraphSAGE_DGL(
        features.shape[1], 64, n_classes, num_layers, use_uva).to('cuda')
    sampler = None
    if config['sample_mode'] == 'ad-hoc':
        sampler = DGLNeighborSampler([25, 10, 10])
    else:
        sampler = DGLNeighborSampler_finegrained([25, 10, 10])
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=batch_size,
                                  shuffle=True,  drop_last=False, num_workers=config['num_workers'], device='cuda')
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=batch_size, shuffle=True,
                                drop_last=False, num_workers=config['num_workers'], device='cuda')

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sample_list = []
    epoch_list = []
    mem_list = []
    feature_loading_list = []
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(config['num_epoch']):
        epoch_feature_loading = 0
        sampling_time = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        model.train()
        tic = time.time()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, blocks) in enumerate(tq):
                torch.cuda.synchronize()
                sampling_time += time.time()-tic

                temp = time.time()
                if use_uva:
                    x = gather_pinned_tensor_rows(
                        features, input_nodes.to('cuda'))
                    y = gather_pinned_tensor_rows(
                        labels, output_nodes.to('cuda'))
                else:
                    x = features[input_nodes]
                    y = labels[output_nodes]
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - temp

                y_hat = model(blocks, x)
                is_labeled = y == y
                y = y[is_labeled].long()
                y_hat = y_hat[is_labeled]
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = compute_acc(y_hat, y)
                tq.set_postfix({'loss': '%.06f' % loss.item(),
                                'acc': '%.03f' % acc.item()})
                tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        tic = time.time()
        with torch.no_grad():
            with tqdm.tqdm(val_dataloader) as tq:
                for it, (input_nodes, output_nodes, blocks) in enumerate(tq):
                    torch.cuda.synchronize()
                    sampling_time += time.time() - tic

                    temp = time.time()
                    if use_uva:
                        x = gather_pinned_tensor_rows(
                            features, input_nodes.to('cuda'))
                        y = gather_pinned_tensor_rows(
                            labels, output_nodes.to('cuda'))
                    else:
                        x = features[input_nodes]
                        y = labels[input_nodes]
                    torch.cuda.synchronize()
                    epoch_feature_loading += time.time() - temp

                    y_pred = model(blocks, x)
                    val_pred.append(y_pred)
                    val_labels.append(y)
                    tic = time.time()
           # acc = compute_acc(val_pred,val_labels)

        torch.cuda.synchronize()
        epoch_list.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        sample_list.append(sampling_time)
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_list[-1], sample_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[1:]))
    print('Average epoch sampling time:', np.mean(sample_list[1:]))
    print('Average epoch end2end time:', np.mean(epoch_list[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))


if __name__ == '__main__':
    config = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1024,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=5,
                        help="numbers of epoch in training")
    parser.add_argument("--sample-mode", default='ad-hoc', choices=['ad-hoc', 'fine-grained'],
                        help="sample mode")
    args = parser.parse_args()
    config['device'] = args.device
    config['use_uva'] = args.use_uva
    config['dataset'] = args.dataset
    config['batch_size'] = args.batchsize
    config['num_workers'] = args.num_workers
    config['num_epoch'] = args.num_epoch
    config['sample_mode'] = args.sample_mode
    print(config)
    print('Loading data')
    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    if args.device != 'cuda':
        config['feat_device'] = 'cpu'
    else:
        config['feat_device'] = 'cuda'
    train_dgl(dataset, config)
