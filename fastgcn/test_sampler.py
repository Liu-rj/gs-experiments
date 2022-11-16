import torch
import dgl
import time
import numpy as np
import gs
from gs.utils import SeedGenerator
import numba
from numba.core import types
from numba.typed import Dict
from dgl.utils import gather_pinned_tensor_rows
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def load_ogb(name):
    data = DglNodePropPredDataset(name=name, root="../datasets")
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


@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai


def dgl_sampler_local_id(g, seeds, fanouts, probs):
    # torch.cuda.nvtx.range_push('dgl sampler')
    for fanout in fanouts:
        insg = dgl.in_subgraph(g, seeds)
        insg = dgl.compact_graphs(insg, seeds)
        cand_nodes = insg.ndata[dgl.NID]
        # torch.cuda.nvtx.range_push('id mapping')
        neighbor_nodes_idx = torch.multinomial(
            probs[cand_nodes], fanout, replacement=False).cpu().numpy()
        seeds_idx = find_indices_in(
            seeds.cpu().numpy(), cand_nodes.cpu().numpy())
        neighbor_nodes_idx = np.union1d(neighbor_nodes_idx, seeds_idx)
        seeds_local_idx = torch.from_numpy(
            find_indices_in(seeds_idx, neighbor_nodes_idx)).cuda()
        neighbor_nodes_idx = torch.from_numpy(neighbor_nodes_idx).cuda()
        # torch.cuda.nvtx.range_pop()
        sg = insg.subgraph(neighbor_nodes_idx)
        seeds = insg.ndata[dgl.NID][sg.ndata[dgl.NID]]
    # torch.cuda.nvtx.range_pop()
    return 1


def dgl_sampler(g, seed_nodes, fanouts, probs, use_uva):
    torch.cuda.nvtx.range_push('dgl sampler')
    blocks = []
    for fanout in fanouts:
        torch.cuda.nvtx.range_push('in subgraph')
        subg = dgl.in_subgraph(g, seed_nodes)  # coo
        torch.cuda.nvtx.range_pop()
        edges = subg.edges()
        nodes = torch.unique(edges[0])  # coo row
        num_pick = np.min([nodes.shape[0], fanout])
        node_probs = probs[nodes]
        torch.cuda.nvtx.range_push('multinomial')
        idx = torch.multinomial(node_probs, num_pick, replacement=False)  # gpu
        torch.cuda.nvtx.range_pop()
        selected = nodes[idx]  # gpu
        torch.cuda.nvtx.range_push('out subgraph')
        subg = dgl.out_subgraph(subg, selected)
        torch.cuda.nvtx.range_pop()
        block = dgl.to_block(subg, seed_nodes)
        seed_nodes = block.srcdata[dgl.NID]
        blocks.insert(0, block)
    torch.cuda.nvtx.range_pop()
    return blocks


def matrix_sampler(A: gs.Matrix, seeds, fanouts, probs, use_uva):
    # torch.cuda.nvtx.range_push('matrix sampler')
    blocks = []
    for fanout in fanouts:
        subA = A[:, seeds]
        row_indices = subA.row_ids()
        node_probs = probs[row_indices]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, node_probs, fanout, False)
        subA = subA[selected, :]
        block = subA.to_dgl_block()
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    # torch.cuda.nvtx.range_pop()
    return 1


def bench(name, func, graph, fanouts, probs, use_uva, iters, node_idx):
    time_list = []
    mem_list = []
    seedloader = SeedGenerator(
        node_idx, batch_size=2048, shuffle=False, drop_last=False)
    torch.cuda.reset_peak_memory_stats()
    graph_storage = torch.cuda.max_memory_allocated()
    print('Static CUDA Storage:', graph_storage / (1024 * 1024 * 1024), 'GB')
    for i in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.time()

        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            ret = func(graph, seeds, fanouts, probs, use_uva)
            if it == 500:
                break

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
        mem_list.append((torch.cuda.max_memory_allocated() - graph_storage) /
                        (1024 * 1024 * 1024))
        print("Sampling Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(
            time_list[-1], mem_list[-1]))

    print(name, "sampling AVG:",
          np.mean(time_list[1:]), " s.")
    print(name, "gpu mem peak AVG:",
          np.mean(mem_list[1:]), " GB.")


use_uva = True
device = torch.device('cuda:%d' % 0)
dataset = load_ogb('ogbn-papers100M')
# dataset = load_ogb('ogbn-products')
dgl_graph = dataset[0]
g = dgl_graph.long()
probs = g.out_degrees().float().cuda()
csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')

# probs = probs.pin_memory()
csc_indptr = csc_indptr.pin_memory()
csc_indices = csc_indices.pin_memory()
# g = g.to("cuda")
# probs = probs.cuda()

matrix = gs.Matrix(gs.Graph(False))
matrix._graph._CAPI_load_csc(csc_indptr, csc_indices)
nodes = g.nodes().cuda()
g.pin_memory_()
print(g)
print('DGL graph g', g.formats())


# bench('DGL Uniform FastGCN', dgl_sampler, g,
#       [2000, 2000], None, iters=10, node_idx=nodes)
bench('DGL Biased FastGCN', dgl_sampler, g,
      [2000, 2000], probs, use_uva, iters=5, node_idx=nodes)

# bench('DGL Local Graph Biased FastGCN', dgl_sampler_local_id, g,
#       [2000, 2000], probs, iters=2, node_idx=nodes)

# bench('Matrix Uniform FastGCN', matrix_sampler, matrix,
#       [2000, 2000], None, iters=10, node_idx=nodes)
bench('Matrix Biased FastGCN', matrix_sampler, matrix,
      [2000, 2000], probs, use_uva, iters=5, node_idx=nodes)
