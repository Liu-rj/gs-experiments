import torch
from gs.utils import load_graph
import dgl
import dgl.function as fn
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


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {
                      'w': edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])})
        return g.edata['w']


def dgl_sampler_local_id(g, seeds, fanouts):
    W = g.edata['weight']
    for fanout in fanouts:
        insg = dgl.in_subgraph(g, seeds)
        insg = dgl.compact_graphs(insg, seeds)
        out_frontier = dgl.reverse(insg, copy_edata=True)
        weight = W[out_frontier.edata[dgl.EID]]
        prob = dgl.ops.copy_e_sum(out_frontier, weight ** 2)
        cand_nodes = insg.ndata[dgl.NID]
        neighbor_nodes_idx = torch.multinomial(
            prob, fanout, replacement=False).cpu().numpy()
        seed_nodes_idx = find_indices_in(
            seeds.cpu().numpy(), cand_nodes.cpu().numpy())
        neighbor_nodes_idx = np.union1d(neighbor_nodes_idx, seed_nodes_idx)
        seed_nodes_local_idx = torch.from_numpy(
            find_indices_in(seed_nodes_idx, neighbor_nodes_idx)).cuda()
        neighbor_nodes_idx = torch.from_numpy(neighbor_nodes_idx).cuda()
        sg = insg.subgraph(neighbor_nodes_idx)
        P = prob[neighbor_nodes_idx]
        weight = W[insg.edata[dgl.EID]][sg.edata[dgl.EID]]
        W_tilde = dgl.ops.e_div_u(sg, weight, P)
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)
        W_tilde = dgl.ops.e_div_v(sg, W_tilde, W_tilde_sum)
        sg.edata['w'] = W_tilde
        seeds = insg.ndata[dgl.NID][sg.ndata[dgl.NID]]


def dgl_sampler(g, seeds, fanouts):
    W = g.edata['weight']
    for fanout in fanouts:
        subg = dgl.in_subgraph(g, seeds)
        # layer-wise sample
        reversed_subg = dgl.reverse(subg, copy_edata=True)
        weight = W[reversed_subg.edata[dgl.EID]]
        probs = dgl.ops.copy_e_sum(reversed_subg, weight ** 2)
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            g.nodes(), probs, fanout, False)
        ################
        selected = torch.cat((seeds, selected)).unique()
        subg = dgl.out_subgraph(subg, selected)
        weight = weight[subg.edata[dgl.EID]]
        W_tilde = dgl.ops.e_div_u(subg, weight, probs)
        W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
        W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
        subg.edata['weight'] = W_tilde
        seeds = selected
    return 1


# def dgl_sampler(g, seeds, fanouts):
#     torch.cuda.nvtx.range_push('dgl sampler')
#     W = g.edata['weight']
#     for fanout in fanouts:
#         torch.cuda.nvtx.range_push('col slicing')
#         subg = dgl.in_subgraph(g, seeds)
#         # layer-wise sample
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('p2sum')
#         reversed_subg = dgl.reverse(subg, copy_edata=True)
#         weight = W[reversed_subg.edata[dgl.EID]]
#         probs = dgl.ops.copy_e_sum(reversed_subg, weight ** 2)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('list sample')
#         selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
#             g.nodes(), probs, fanout, False)
#         ################
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('self-loop')
#         selected = torch.cat((seeds, selected)).unique()
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('row slicing')
#         subg = dgl.out_subgraph(subg, selected)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('divide')
#         weight = weight[subg.edata[dgl.EID]]
#         W_tilde = dgl.ops.e_div_u(subg, weight, probs)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('normalize')
#         W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
#         W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
#         subg.edata['weight'] = W_tilde
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('all nodes')
#         seeds = selected
#         torch.cuda.nvtx.range_pop()
#     torch.cuda.nvtx.range_pop()
#     return 1


def matrix_sampler(P: gs.Matrix, seeds, fanouts):
    for fanout in fanouts:
        U = P[:, seeds]
        prob = U.sum(axis=1, powk=2)
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            U.row_ids(unique=False), prob, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subU = U[nodes, :]
        subU = subU.divide(prob[nodes], axis=1)
        subU = subU.normalize(axis=0)
        seeds = nodes
    return 1


# def matrix_sampler(P: gs.Matrix, seeds, fanouts):
#     torch.cuda.nvtx.range_push('matrix sampler')
#     for fanout in fanouts:
#         torch.cuda.nvtx.range_push('col slicing')
#         U = P[:, seeds]
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('p2sum')
#         prob = U.sum(axis=1, powk=2)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('list sample')
#         selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
#             U.row_ids(unique=False), prob, fanout, False)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('self-loop')
#         nodes = torch.cat((seeds, selected)).unique()  # add self-loop
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('row slicing')
#         subU = U[nodes, :]
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('divide')
#         subU = subU.divide(prob[nodes], axis=1)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('normalize')
#         subU = subU.normalize(axis=0)
#         torch.cuda.nvtx.range_pop()
#         torch.cuda.nvtx.range_push('all nodes')
#         seeds = nodes
#         torch.cuda.nvtx.range_pop()
#     torch.cuda.nvtx.range_pop()
#     return 1


def bench(name, func, graph, fanouts, iters, node_idx):
    time_list = []
    mem_list = []
    seedloader = SeedGenerator(
        node_idx, batch_size=1024, shuffle=False, drop_last=False)
    torch.cuda.reset_peak_memory_stats()
    graph_storage = torch.cuda.max_memory_allocated()
    print('Raw Graph Storage:', graph_storage / (1024 * 1024 * 1024), 'GB')
    for i in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.time()

        for it, seeds in enumerate(seedloader):
            ret = func(graph, seeds, fanouts)
            if it == 100:
                break

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
        mem_list.append((torch.cuda.max_memory_allocated() - graph_storage) /
                        (1024 * 1024 * 1024))
        print("Sampling Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(
            time_list[-1], mem_list[-1]))

    print(name, "sampling AVG:",
          np.mean(time_list[3:]), " s.")
    print(name, "gpu mem peak AVG:",
          np.mean(mem_list[3:]), " GB.")


device = torch.device('cuda:%d' % 0)
dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.to("cuda")
g.edata['weight'] = normalized_laplacian_edata(g)
matrix = gs.Matrix(gs.Graph(False))
matrix.load_dgl_graph(g)
D_in = matrix.sum(axis=0)
D_out = matrix.sum(axis=1)
matrix = matrix.divide(D_out.sqrt(), axis=1).divide(D_in.sqrt(), axis=0)
nodes = g.nodes()
print(g)
print('DGL graph formats:', g.formats())


bench('DGL LADIES', dgl_sampler, g,
      [2000, 2000], iters=10, node_idx=nodes)
bench('DGL Local Graph LADIES', dgl_sampler_local_id, g,
      [2000, 2000], iters=10, node_idx=nodes)
bench('Matrix LADIES', matrix_sampler, matrix,
      [2000, 2000], iters=10, node_idx=nodes)
