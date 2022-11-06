import torch
import torch.nn.functional as F
import dgl
from dgl.data import RedditDataset
from dgl.dataloading import DataLoader
import tqdm
from model import *
from sampler import *
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.utils import gather_pinned_tensor_rows
import numpy as np


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args['device']
    use_uva = args['use_uva']

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g, train_nid, val_nid = g.to(device), train_nid.to(
        device), val_nid.to(device)
    g.edata['weight'] = normalized_laplacian_edata(g)
    g.ndata['nodes'] = g.nodes()
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        g.edata['weight'] = g.edata['weight'].pin_memory()
        g.ndata['nodes'] = g.ndata['nodes'].pin_memory()
    else:
        features, labels = features.to(device), labels.to(device)

    num_nodes = args['num_nodes']
    sampler = LADIESSampler(num_nodes, weight='weight',
                            out_weight='w', replace=False, use_uva=use_uva)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'], use_uva=use_uva)
    val_dataloader = DataLoader(
        g,
        val_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'], use_uva=use_uva)

    model = Model(features.shape[1],
                  args['hidden_dim'], n_classes, 2).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')

    for epoch in range(args['num_epochs']):
        epoch_feature_loading = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, seeds, blocks) in enumerate(tq):
                torch.cuda.synchronize()
                temp = time.time()
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, seeds)
                else:
                    batch_inputs = features[input_nodes]
                    batch_labels = labels[seeds]
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - temp

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                acc = compute_acc(batch_pred, batch_labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tq.set_postfix({'loss': '%.06f' % loss.item(),
                               'acc': '%.03f' % acc.item()})

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            with tqdm.tqdm(val_dataloader) as tq:
                for step, (input_nodes, seeds, blocks) in enumerate(tq):
                    torch.cuda.synchronize()
                    temp = time.time()
                    if use_uva:
                        batch_inputs = gather_pinned_tensor_rows(
                            features, input_nodes)
                        batch_labels = gather_pinned_tensor_rows(labels, seeds)
                    else:
                        batch_inputs = features[input_nodes]
                        batch_labels = labels[seeds]
                    torch.cuda.synchronize()
                    epoch_feature_loading += time.time() - temp

                    batch_pred = model(blocks, batch_inputs)
                    is_labeled = batch_labels == batch_labels
                    batch_labels = batch_labels[is_labeled].long()
                    batch_pred = batch_pred[is_labeled]
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)
            acc = compute_acc(torch.cat(val_pred, 0),
                              torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        time_list.append(sampler.sampling_time)
        sampler.sampling_time = 0
        mem_list.append((torch.cuda.max_memory_reserved() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_time[-1], time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))


def load_reddit():
    data = RedditDataset(self_loop=True)
    g = data[0].long()
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_ogb(name):
    data = DglNodePropPredDataset(name=name, root="../../datasets")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g = g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


if __name__ == '__main__':
    config = {
        'num_epochs': 5,
        'hidden_dim': 64}

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--samples", default='2000,2000',
                        help="sample size for each layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    args = parser.parse_args()
    config['device'] = args.device
    config['use_uva'] = args.use_uva
    config['dataset'] = args.dataset
    config['batch_size'] = args.batchsize
    config['num_nodes'] = [int(x.strip()) for x in args.samples.split(',')]
    config['num_workers'] = args.num_workers
    print(config)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    train(dataset, config)
