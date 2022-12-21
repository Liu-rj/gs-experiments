import dgl
import gs
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import time
import numpy as np

_DCSR = 16
_DCSC = 8
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

for num_seeds in [1000, 10000, 100000, 500000, 1000000]:
    seeds = node_seeds[:num_seeds].cuda()
    time_list = []

    # DGL SpMM
    subgraph = dgl.in_subgraph(graph, seeds).to('cuda')
    num_edges = subgraph.num_edges()
    weight = torch.ones(num_edges, dtype=torch.float32).cuda()

    subgraph = subgraph.formats("coo")
    time_list.clear()
    for i in range(100):
        tic = time.time()
        dgl.ops.copy_e_sum(subgraph, weight)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print(" DGL", "COO", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    subgraph = subgraph.formats("csc")
    time_list.clear()
    for i in range(100):
        tic = time.time()
        dgl.ops.copy_e_sum(subgraph, weight)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print(" DGL", "CSC", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    # Full
    sub_full_m = full_m._CAPI_full_slicing(seeds, 0, _CSC)
    sub_full_m._CAPI_set_data(weight.clone())
    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_full_m._CAPI_full_sum(0, 1, _COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "COO", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    sub_full_m._CAPI_set_data(weight.clone())
    for i in range(100):
        tic = time.time()
        sub_full_m._CAPI_full_sum(0, 1, _CSC)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "CSC", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    # DCSR
    sub_m = m._CAPI_slicing(seeds, 0, _CSC, _CSC)
    sub_m._CAPI_set_data(weight.clone())
    time_list.clear()
    for i in range(100):
        tic = time.time()
        sub_m._CAPI_sum(0, 1, _COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "COO", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))

    sub_m._CAPI_set_data(weight.clone())
    for i in range(100):
        tic = time.time()
        sub_m._CAPI_sum(0, 1, _CSC)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "CSC", num_seeds, num_edges, 1000 * np.mean(time_list[10:]))
