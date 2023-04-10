import gs
from gs.utils import load_graph, SeedGenerator
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse


def sample_w_o_relabel(A: gs.Matrix, seeds, probs, fanouts):
    graph = A._graph
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        row_indices = torch.unique(subg._CAPI_get_coo_rows(False))
        node_probs = probs[row_indices]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, node_probs, fanout, False)
        subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, ret


def sample_w_relabel(A: gs.Matrix, seeds, probs, fanouts):
    graph = A._graph
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, True)
        node_probs = probs[subg._CAPI_get_rows()]
        selected = torch.multinomial(node_probs, fanout, replacement=False)
        subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, ret


def benchmark_w_o_relabel(args, matrix, nid, fanouts, n_epoch, probs):
    print('####################################################w/o relabel')
    seedloader = SeedGenerator(
        nid, batch_size=args.batchsize, shuffle=True, drop_last=False)

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            input_nodes, output_nodes, blocks = sample_w_o_relabel(
                matrix, seeds, probs, fanouts)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print('Average epoch sampling time:', np.mean(epoch_time[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))
    print('####################################################END')


def benchmark_w_relabel(args, matrix, nid, fanouts, n_epoch, probs):
    print('####################################################w relabel')
    seedloader = SeedGenerator(
        nid, batch_size=args.batchsize, shuffle=True, drop_last=False)

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            input_nodes, output_nodes, blocks = sample_w_relabel(
                matrix, seeds, probs, fanouts)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print('Average epoch sampling time:', np.mean(epoch_time[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))
    print('####################################################END')


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(',')]

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.long()
    train_nid = splitted_idx['train']
    g, train_nid = g.to(device), train_nid.to('cuda')
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    probs = g.out_degrees().float().to('cuda')
    if use_uva and device == 'cpu':
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        # probs = probs.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    n_epoch = 6
    benchmark_w_o_relabel(args, m, train_nid, fanouts, n_epoch, probs)
    benchmark_w_relabel(args, m, train_nid, fanouts, n_epoch, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', type=bool, default=False,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='ogbn-products',
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--samples", default='400,400',
                        help="sample size for each layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith('ogbn'):
        dataset = load_graph.load_ogb(args.dataset, '/home/ubuntu/dataset')
    elif args.dataset == 'livejournal':
        dataset = load_graph.load_dglgraph(
            '/home/ubuntu/dataset/livejournal/livejournal.bin')
    elif args.dataset == 'friendster':
        dataset = load_graph.load_dglgraph(
            '/home/ubuntu/dataset/friendster/friendster.bin')
    else:
        raise NotImplementedError
    print(dataset[0])
    train(dataset, args)
