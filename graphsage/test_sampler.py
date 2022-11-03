import torch
from gs.utils import load_graph
import dgl
from dgl.transforms.functional import to_block
import time
import numpy as np
import gs
from gs.utils import SeedGenerator
from ogb.nodeproppred import DglNodePropPredDataset


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def load_ogbn_papers100M():
    data = DglNodePropPredDataset(
        name="ogbn-products", root="../datasets")
    g, _ = data[0]
    g.ndata.clear()
    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print(g)
    return g


def dgl_sampler(g, seeds, fanouts):
    # torch.cuda.nvtx.range_push('dgl sampler')
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('in subgraph')
        subg = dgl.in_subgraph(g, seeds)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sample neighbors')
        subg = subg.sample_neighbors(seeds, fanout)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('edges')
        edges = subg.edges()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('unique_cat')
        seeds = torch.unique(torch.cat(edges))
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def matrix_sampler_nonfused(A: gs.Matrix, seeds, fanouts):
    # torch.cuda.nvtx.range_push('matrix nonfused sampler')
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('slicing')
        subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sampling')
        subA = subA.columnwise_sampling(fanout, replace=False)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all indices')
        seeds = subA.all_indices()
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def matrix_sampler_fused(A: gs.Matrix, seeds, fanouts):
    # torch.cuda.nvtx.range_push('matrix fused sampler')
    for fanout in fanouts:
        # torch.cuda.nvtx.range_push('slice_sample')
        subA = gs.Matrix(
            A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all indices')
        seeds = subA.all_indices()
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def bench(name, func, graph, fanouts, iters, node_idx):
    time_list = []
    mem_list = []
    seedloader = SeedGenerator(
        node_idx, batch_size=1024, shuffle=True, drop_last=False)
    for i in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.time()

        for it, seeds in enumerate(seedloader):
            ret = func(graph, seeds, fanouts)

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
        mem_list.append((torch.cuda.max_memory_allocated() - graph_storage) /
                        (1024 * 1024 * 1024))
        print("Sampling Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(
            time_list[-1], mem_list[-1]))

    print(name, "graphsage sampling AVG:",
          np.mean(time_list[3:]), " s.")
    print(name, "graphsage gpu mem peak AVG:",
          np.mean(mem_list[3:]), " GB.")


device = torch.device('cuda:%d' % 0)
dgl_graph = load_ogbn_papers100M()
g = dgl_graph.long()
g = g.to("cuda")
matrix = gs.Matrix(gs.Graph(False))
matrix.load_dgl_graph(g)
nodes = g.nodes()
torch.cuda.reset_peak_memory_stats()
graph_storage = torch.cuda.max_memory_allocated()
print('Raw Graph Storage:', graph_storage / (1024 * 1024 * 1024), 'GB')
print(g)
print('DGL graph g', g.formats())


bench('DGL', dgl_sampler, g, [25, 10], iters=10, node_idx=nodes)
bench('Matrix Non-fused', matrix_sampler_nonfused, matrix,
      [25, 10], iters=10, node_idx=nodes)
# bench('Matrix Fused', matrix_sampler_fused, matrix,
#       [25, 10], iters=10, node_idx=nodes)
