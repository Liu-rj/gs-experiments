import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse


def sample_w_o_relabel(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    blocks = []
    for fanout in fanouts:
        subg = graph._CAPI_fused_columnwise_slicing_sampling(
            seeds, fanout, False)
        all_nodes = torch.cat([seeds, subg._CAPI_get_coo_rows(False)])
        seeds = torch.ops.gs_ops._CAPI_unique(all_nodes)
    subg = graph._CAPI_fusion_slicing(seeds, seeds)
    # subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, False)
    # subg = subg._CAPI_slicing(seeds, 1, gs._CSC, gs._CSC, False)
    unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
    input_node = seeds
    return input_node, output_node, blocks


def benchmark(args, matrix, nid, fanouts, n_epoch, sampler):
    print('####################################################{}'.format(
        sampler.__name__))
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
            input_nodes, output_nodes, blocks = sampler(
                matrix, fanouts, seeds)

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
    # weight = normalized_laplacian_edata(g)
    weight = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)
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
