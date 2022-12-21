import dgl
import gs
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import time
import numpy as np

_CSR = 4
_CSC = 2
_COO = 1

torch.manual_seed(1)

data = DglNodePropPredDataset(name='ogbn-products', root="../../datasets")
graph = data[0][0]
graph.ndata.clear()
graph.edata.clear()

graph = graph.to('cuda')

indptr, indices, _ = graph.adj_sparse('csc')
indptr = indptr.cuda()
indices = indices.cuda()

m = gs.Graph(False)
m._CAPI_load_csc(indptr, indices)

full_m = gs.Graph(False)
full_m._CAPI_full_load_csc(indptr, indices)

node_seeds = torch.randperm(graph.num_nodes())

for num_seeds in [1000]:  #, 10000, 100000, 500000, 1000000]:
    seeds = node_seeds[:num_seeds].cuda()
    time_list = []

    # DGL SpMM
    subgraph = dgl.in_subgraph(graph, seeds)
    for i in subgraph.adj_sparse('coo'):
        print(i)
    num_edges = subgraph.num_edges()
    time_list.clear()
    for i in range(100):
        tic = time.time()
        dgl.out_subgraph(subgraph, seeds)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)

    print(" DGL", "CSR", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    # Full
    sub_full_m = full_m._CAPI_full_slicing(seeds, 0, _CSC)
    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_full_m._CAPI_full_slicing(seeds, 1, _COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "COO", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_full_m._CAPI_full_slicing(seeds, 1, _CSR)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "CSR", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    for i in range(100):
        tic = time.time()
        sub_full_m._CAPI_full_slicing(seeds, 1, _CSC)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "CSC", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    # DCSR
    sub_m = m._CAPI_slicing(seeds, 0, _CSC, _CSC)
    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_m._CAPI_slicing(seeds, 1, _COO, _COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "COO", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_m._CAPI_slicing(seeds, 1, _CSR, _CSR)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "CSR", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    for i in range(100):
        tic = time.time()
        sub_m._CAPI_slicing(seeds, 1, _CSC, _CSC)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "CSC", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))
