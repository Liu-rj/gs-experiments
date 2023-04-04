import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse


def sample_w_o_relabel(A: gs.Matrix, seeds, seeds_ptr, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    num_batches = seeds_ptr.numel() - 1
    for fanout in fanouts:
        subg, coo_ptr = graph._CAPI_batch_slicing(
            seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True)
        p = subg._CAPI_sum(1, 2, gs._COO)
        p = p.sqrt()
        colid, rowid, data, coo_row, coo_col = subg._CAPI_coo_metadata()
        batch_seeds = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        coo_row = torch.ops.gs_ops.BatchDecode(coo_row, encoding_size)
        batch_coo_rows = torch.ops.gs_ops.SplitByOffset(coo_row, coo_ptr)
        batch_coo_cols = torch.ops.gs_ops.SplitByOffset(coo_col, coo_ptr)
        batch_datas = torch.ops.gs_ops.SplitByOffset(data, coo_ptr)
        batch_ps = torch.split(p, graph._CAPI_get_num_rows())
        list_unique_tensors = []
        next_seeds_ptr = torch.zeros_like(seeds_ptr)
        for i, batch_seed in enumerate(batch_seeds):
            subg = gs.Graph(True)
            subg._CAPI_set_metadata(
                batch_seed, None, batch_seed.numel(), graph._CAPI_get_num_rows())
            subg._CAPI_set_coo(
                batch_coo_rows[i], batch_coo_cols[i] - seeds_ptr[i], None, False, True)
            subg._CAPI_set_data(batch_datas[i])

            row_indices = torch.unique(subg._CAPI_get_coo_rows(False))
            if use_uva:
                node_feats_u = gather_pinned_tensor_rows(features, row_indices)
                node_feats_v = gather_pinned_tensor_rows(features, seeds)
            else:
                node_feats_u = features[row_indices]
                node_feats_v = features[seeds]
            h_u = node_feats_u @ W[:, 0]
            h_v = node_feats_v @ W[:, 1]
            h_v_sum = torch.sum(h_v)
            attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
            g_u = torch.flatten(F.relu(h_u) + 1)

            q = F.normalize(batch_ps[i][row_indices]
                            * attention * g_u, p=1.0, dim=0)

            selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
                row_indices, q, fanout, False)

            subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
            W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u[idx], h_v, gs._COO)
            W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
            W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q[idx], gs._COO)
            subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
            list_unique_tensors.append(unique_tensor)
            next_seeds_ptr[i + 1] = unique_tensor.numel() + next_seeds_ptr[i]
        seeds = torch.cat(list_unique_tensors)
        seeds_ptr = next_seeds_ptr
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def sample_w_relabel(A: gs.Matrix, seeds, seeds_ptr, features, W, fanouts, use_uva):
    graph = A._graph
    output_nodes = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    num_batches = seeds_ptr.numel() - 1
    for it, fanout in enumerate(fanouts):
        subg, coo_ptr = graph._CAPI_batch_slicing(
            seeds, seeds_ptr, 0, gs._CSC, gs._COO, True, True)
        p = subg._CAPI_sum(1, 2, gs._COO)
        p = p.sqrt()
        colid, rowid, data, coo_row, coo_col = subg._CAPI_coo_metadata()
        row_ptr, decoded_row = torch.ops.gs_ops.GetBatchOffsets(
            rowid, num_batches, encoding_size)
        batch_seeds = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        batch_rows = torch.ops.gs_ops.SplitByOffset(decoded_row, row_ptr)
        batch_coo_rows = torch.ops.gs_ops.SplitByOffset(coo_row, coo_ptr)
        batch_coo_cols = torch.ops.gs_ops.SplitByOffset(coo_col, coo_ptr)
        batch_datas = torch.ops.gs_ops.SplitByOffset(data, coo_ptr)
        batch_ps = torch.ops.gs_ops.SplitByOffset(p, row_ptr)
        list_unique_tensors = []
        next_seeds_ptr = torch.zeros_like(seeds_ptr)
        for i, batch_seed in enumerate(batch_seeds):
            subg = gs.Graph(True)
            subg._CAPI_set_metadata(
                batch_seed, batch_rows[i], batch_seed.numel(), batch_rows[i].numel())
            subg._CAPI_set_coo(
                batch_coo_rows[i] - row_ptr[i], batch_coo_cols[i] - seeds_ptr[i], None, False, True)
            subg._CAPI_set_data(batch_datas[i])

            row_indices = subg._CAPI_get_rows()
            num_pick = np.min([row_indices.numel(), fanout])
            if use_uva:
                node_feats_u = gather_pinned_tensor_rows(features, row_indices)
                node_feats_v = gather_pinned_tensor_rows(features, seeds)
            else:
                node_feats_u = features[row_indices]
                node_feats_v = features[seeds]
            h_u = node_feats_u @ W[:, 0]
            h_v = node_feats_v @ W[:, 1]
            h_v_sum = torch.sum(h_v)
            attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
            g_u = torch.flatten(F.relu(h_u) + 1)

            q = F.normalize(batch_ps[i] * attention * g_u, p=1.0, dim=0)

            selected = torch.multinomial(q, num_pick, replacement=False)

            subg = subg._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
            W_tilde = gs.ops.u_add_v(gs.Matrix(subg), h_u, h_v, gs._COO)
            W_tilde = (F.relu(W_tilde) + 1) / selected.numel()
            W_tilde = gs.ops.e_div_u(gs.Matrix(subg), W_tilde, q, gs._COO)
            subg._CAPI_set_data(W_tilde * subg._CAPI_get_data('default'))

            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
            list_unique_tensors.append(unique_tensor)
            next_seeds_ptr[i + 1] = unique_tensor.numel() + next_seeds_ptr[i]
        seeds = torch.cat(list_unique_tensors)
        seeds_ptr = next_seeds_ptr
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def benchmark(args, matrix, nid, fanouts, n_epoch, features, W, sampler):
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
                matrix, seeds, seeds_ptr, features, W, fanouts, args.use_uva)

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
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W = torch.nn.init.xavier_normal_(
        torch.Tensor(features.shape[1], 2)).to('cuda')
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    weight = weight[edge_ids]
    if args.use_uva and device == 'cpu':
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        weight, features = weight.pin_memory(), features.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    m._graph._CAPI_set_data(weight)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    n_epoch = 6
    # benchmark(args, m, train_nid, fanouts, n_epoch,
    #           features, W, sample_w_o_relabel)
    benchmark(args, m, train_nid, fanouts, n_epoch,
              features, W, sample_w_relabel)


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
    parser.add_argument("--batching-batchsize", type=int, default=51200,
                        help="batch size for training")
    parser.add_argument("--samples", default='512,512',
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
