import torch
from gs.utils import load_graph
import dgl
from dgl.transforms.functional import to_block
import time
import numpy as np
import gs
from gs.utils import SeedGenerator


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def dgl_sampler(g, seeds, num_steps):
    # torch.cuda.nvtx.range_push('dgl sampler')
    paths = [seeds]
    for step in range(num_steps):
        # torch.cuda.nvtx.range_push('in subgraph')
        subg = dgl.in_subgraph(g, seeds)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sample neighbors')
        subg = subg.sample_neighbors(seeds, 1)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('edges')
        edges = subg.edges()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('unique_cat')
        seeds = edges[1]
        paths.append(seeds)
    sample_nodes = torch.stack(paths).flatten()
    sample_nodes = sample_nodes[sample_nodes != -1]
    sample_nodes = torch.unique(sample_nodes)
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def matrix_sampler_nonfused(A: gs.Matrix, seeds, num_steps):
    # torch.cuda.nvtx.range_push('matrix nonfused sampler')
    paths = [seeds]
    for step in range(num_steps):
        # torch.cuda.nvtx.range_push('slicing')
        subA = A[:, seeds]
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('sampling')
        subA = subA.columnwise_sampling(1, replace=False)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all indices')
        seeds = subA.row_indices()
        paths.append(seeds)
    sample_nodes = torch.stack(paths).flatten()
    sample_nodes = sample_nodes[sample_nodes != -1]
    sample_nodes = torch.unique(sample_nodes)
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def matrix_sampler_fused(A: gs.Matrix, seeds, num_steps):
    # torch.cuda.nvtx.range_push('matrix fused sampler')
    paths = [seeds]
    for step in range(num_steps):
        # torch.cuda.nvtx.range_push('slice_sample')
        subA = gs.Matrix(
            A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, 1, False))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('all indices')
        seeds = subA.row_indices()
        paths.append(seeds)
    sample_nodes = torch.stack(paths).flatten()
    sample_nodes = sample_nodes[sample_nodes != -1]
    sample_nodes = torch.unique(sample_nodes)
    #     torch.cuda.nvtx.range_pop()
    # torch.cuda.nvtx.range_pop()
    return 1


def bench(name, func, graph, num_steps, iters, node_idx):
    time_list = []
    mem_list = []
    seedloader = SeedGenerator(
        node_idx, batch_size=1024, shuffle=True, drop_last=False)
    graph_storage = torch.cuda.memory_allocated()
    print('Raw Graph Storage:', graph_storage / (1024 * 1024 * 1024), 'GB')
    for i in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.time()

        for it, seeds in enumerate(seedloader):
            ret = func(graph, seeds, num_steps)

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
g = dataset[0].long()
g = g.to("cuda")
matrix = gs.Matrix(gs.Graph(False))
matrix.load_dgl_graph(g)
nodes = g.nodes()
print(g)
print('DGL graph g', g.formats())


bench('DGL random walk', dgl_sampler, g, 4, iters=10, node_idx=nodes)
bench('Matrix random walk Non-fused', matrix_sampler_nonfused, matrix,
      4, iters=10, node_idx=nodes)
bench('Matrix random walk Fused', matrix_sampler_fused, matrix,
      4, iters=10, node_idx=nodes)
