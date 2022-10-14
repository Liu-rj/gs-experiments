from random import seed
import gs
from typing import List
import torch
import load_graph
import time
import numpy as np


time_list = []
seeds_num = []


def fastgcn(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    acc_time = 0
    seeds_num.append(seeds.numel())
    for fanout in fanouts:
        # print(seeds.shape)
        torch.cuda.synchronize()
        begin = time.time()
        subA = A[:, seeds]
        selected, _ = torch.ops.gs_ops.list_sampling(subA.row_indices(),
                                                     fanout, False)
        torch.cuda.synchronize()
        end = time.time()
        acc_time += end - begin
        subA = subA[selected, :]
        seeds = subA.all_indices()
        seeds_num.append(seeds.numel())
        ret.append(subA)
    output_node = seeds
    time_list.append(acc_time)
    return input_node, output_node, ret


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
seeds = torch.randint(0, 200000, (1000,)).long().cuda()

# compiled_func = gs.jit.compile(func=fastgcn, args=(m, seeds, [2000, 2000]))
# print(compiled_func.gm)


def bench(func, args):
    for i in range(1):
        # torch.cuda.synchronize()
        # begin = time.time()

        ret = func(*args)

        # torch.cuda.synchronize()
        # end = time.time()

        # time_list.append(end - begin)

    # print("fastgcn sampling AVG:", np.mean(time_list[3:]) * 1000, " ms.")


bench(fastgcn, args=(m, seeds, [200, 200]))
print(seeds_num)