import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse


def sample_w_o_relabel(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    graph = P._graph
    output_node = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_fused_columnwise_slicing_sampling(
            seeds, fanout, False)
        indptr, indices, eid = subg._CAPI_get_csc()
        indices_ptr = indptr[seeds_ptr]
        all_nodes, key, ptr = torch.ops.gs_ops.BatchConcat(
            [seeds, subg._CAPI_get_coo_rows(False)], [seeds_ptr, indices_ptr])
        seeds, seeds_ptr, key = torch.ops.gs_ops.BatchUniqueByKey2(
            all_nodes, ptr, key)
    subg = graph._CAPI_batch_fusion_slicing(seeds, seeds_ptr, seeds, seeds_ptr)
    indptr, indices, eid = subg._CAPI_get_csc()
    indices_ptr = indptr[seeds_ptr]
    unique_tensor, unique_tensor_ptr, indices, indices_ptr = torch.ops.gs_ops.BatchCSRRelabel(
        seeds, seeds_ptr, indices, indices_ptr)
    seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
    unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
    indicest = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)
    blocks.insert(0, (seedst, unit, indicest))
    input_node = seeds
    return input_node, output_node, blocks


def benchmark(args, matrix, nid, fanouts, n_epoch, sampler):
    print('####################################################{}'.format(
        sampler.__name__))
    batch_size = args.batching_batchsize
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(
        num_batches + 1, dtype=torch.int64, device='cuda') * small_batch_size

    seedloader = SeedGenerator(
        nid, batch_size=batch_size, shuffle=True, drop_last=False)

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
            seeds_ptr = orig_seeds_ptr
            if it == len(seedloader) - 1:
                num_batches = int(
                    (seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1,
                                         dtype=torch.int64,
                                         device='cuda') * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            input_nodes, output_nodes, blocks = sampler(
                matrix, fanouts, seeds, seeds_ptr)

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
    if use_uva and device == 'cpu':
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    n_epoch = 6
    benchmark(args, m, train_nid, fanouts, n_epoch, sample_w_o_relabel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', type=bool, default=False,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='ogbn-products',
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--batching-batchsize", type=int, default=5120,
                        help="batch size for training")
    parser.add_argument("--samples", default='10,10',
                        help="sample size for each layer")
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
