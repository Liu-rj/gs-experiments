import gs
from gs.utils import SeedGenerator
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
from load_graph import *
from model import *
from sampler import *


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(',')]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g, train_nid, val_nid = g.to(device), train_nid.to(
        device), val_nid.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    probs = g.out_degrees().float()
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
        probs = probs.pin_memory()
    else:
        features, labels = features.to(device), labels.to(device)
        probs = probs.to(device)
    m = gs.Matrix(gs.Graph(False))
    # m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    m._graph._CAPI_full_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    # compiled_func = gs.jit.compile(
    #     func=fastgcn_sampler, args=(m, torch.Tensor(), fanouts))
    # compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))
    compiled_func = fastgcn_matrix_sampler_with_format_selection_coo_full
    train_seedloader = SeedGenerator(
        train_nid, batch_size=args.batchsize, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_nid, batch_size=args.batchsize, shuffle=True, drop_last=False)
    model = Model(features.shape[1], 64, n_classes, len(fanouts)).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epoch = 5

    sample_time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(n_epoch):
        epoch_feature_loading = 0
        sample_time = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_loss = 0
        torch.cuda.synchronize()
        tic = time.time()
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds = seeds.to('cuda')
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, probs, fanouts)
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            tic = time.time()
            blocks = [block.to('cuda') for block in blocks]
            if use_uva:
                batch_inputs = gather_pinned_tensor_rows(
                    features, input_nodes)
                batch_labels = gather_pinned_tensor_rows(labels, seeds)
            else:
                batch_inputs = features[input_nodes].to('cuda')
                batch_labels = labels[seeds].to('cuda')
            torch.cuda.synchronize()
            epoch_feature_loading += time.time() - tic

            batch_pred = model(blocks, batch_inputs)
            is_labeled = batch_labels == batch_labels
            batch_labels = batch_labels[is_labeled].long()
            batch_pred = batch_pred[is_labeled]
            loss = F.cross_entropy(batch_pred, batch_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            torch.cuda.synchronize()
            tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            torch.cuda.synchronize()
            tic = time.time()
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds = seeds.to('cuda')
                input_nodes, output_nodes, blocks = compiled_func(
                    m, seeds, probs, fanouts)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, seeds)
                else:
                    batch_inputs = features[input_nodes].to('cuda')
                    batch_labels = labels[seeds].to('cuda')
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                val_pred.append(batch_pred)
                val_labels.append(batch_labels)
                torch.cuda.synchronize()
                tic = time.time()

        acc = compute_acc(torch.cat(val_pred, 0),
                          torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        sample_time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, acc, epoch_time[-1], sample_time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(sample_time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))

    # print('Testing...')
    # acc = layerwise_infer(g, test_nid, model,
    #                       batch_size=4096, feat=features, label=labels)
    # print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
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
    print(args)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    train(dataset, args)
