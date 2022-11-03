import torch
from gs.utils import load_graph
import dgl
import time
import numpy as np
import gs
from gs.utils import SeedGenerator
import numba
from numba.core import types
from numba.typed import Dict
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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


def dgl_sampler(g, seeds, fanouts, probs):
    # torch.cuda.nvtx.range_push('dgl sampler')
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('in subgraph')
        subg = dgl.in_subgraph(g, seeds)
        # layer-wise sample
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row ids')
        edges = subg.edges()
        nodes = torch.unique(edges[0])
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('list sample')
        if probs is not None:
            selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
                nodes, probs[nodes], fanout, False)
        else:
            selected, _ = torch.ops.gs_ops.list_sampling(nodes,
                                                         fanout, False)
        selected = torch.cat((seeds, selected)).unique()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('out subgraph')
        ################
        subg = dgl.out_subgraph(subg, selected)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all nodes')
        seeds = selected
        # torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def matrix_sampler(A: gs.Matrix, seeds, fanouts, probs):
    # torch.cuda.nvtx.range_push('matrix sampler')
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('col slice')
        subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row ids')
        row_indices = subA.row_ids()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('list sample')
        if probs is not None:
            selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
                row_indices, probs[row_indices], fanout, False)
        else:
            selected, _ = torch.ops.gs_ops.list_sampling(row_indices,
                                                         fanout, False)
        selected = torch.cat((seeds, selected)).unique()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('row slice')
        subA = subA[selected, :]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all nodes')
        seeds = selected
        # torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def bench(name, func, graph, fanouts, probs, iters, node_idx):
    time_list = []
    mem_list = []
    seedloader = SeedGenerator(
        node_idx, batch_size=1024, shuffle=False, drop_last=False)
    torch.cuda.reset_peak_memory_stats()
    graph_storage = torch.cuda.max_memory_allocated()
    print('Raw Graph Storage:', graph_storage / (1024 * 1024 * 1024), 'GB')
    print(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 'GB')
    for i in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.time()

        for it, seeds in enumerate(seedloader):
            ret = func(graph, seeds, fanouts, probs)

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


device = torch.device('cuda:%d' % 0)
dataset = load_graph.load_ogbn_products()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.to("cuda")
matrix = gs.Matrix(gs.Graph(False))
matrix.load_dgl_graph(g)
nodes = g.nodes()
probs = g.out_degrees().float().cuda()
print(g)
print('DGL graph g', g.formats())


# bench('DGL Uniform FastGCN', dgl_sampler, g,
#       [2000, 2000], None, iters=10, node_idx=nodes)
bench('DGL Biased FastGCN', dgl_sampler, g,
      [2000, 2000], probs, iters=2, node_idx=nodes)

bench('DGL Local Graph Biased FastGCN', dgl_sampler_local_id, g,
      [2000, 2000], probs, iters=2, node_idx=nodes)

# bench('Matrix Uniform FastGCN', matrix_sampler, matrix,
#       [2000, 2000], None, iters=10, node_idx=nodes)
bench('Matrix Biased FastGCN', matrix_sampler, matrix,
      [2000, 2000], probs, iters=2, node_idx=nodes)
