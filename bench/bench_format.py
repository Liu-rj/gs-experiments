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

    # DGL Format Conversion
    subgraph = dgl.in_subgraph(graph, seeds).to('cuda')
    num_edges = subgraph.num_edges()
    subgraph = subgraph.formats("csc")

    time_list.clear()
    for i in range(100):
        subgraph = subgraph.formats(["csc"])
        subgraph = subgraph.formats(["csc", "coo"])
        tic = time.time()
        subgraph.adj_sparse('coo')
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print(" DGL", "CSC2COO", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        subgraph = subgraph.formats(["csc", "coo"])
        subgraph = subgraph.formats(["csc", "coo", "csr"])
        tic = time.time()
        subgraph.adj_sparse('csr')
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print(" DGL", "COO2CSR", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    # Full Format Conversion
    sub_full_m = full_m._CAPI_full_slicing(seeds, 0, _CSC)
    sub_full_m._CAPI_full_create_format(_CSC)
    time_list.clear()
    for i in range(100):
        sub_full_m._CAPI_drop_format(_COO)
        tic = time.time()
        sub_full_m._CAPI_full_create_format(_COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "CSC2COO", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        sub_full_m._CAPI_drop_format(_CSR)
        tic = time.time()
        sub_full_m._CAPI_full_create_format(_CSR)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("Full", "COO2CSR", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    # DCSR
    sub_m = m._CAPI_slicing(seeds, 0, _CSC, _CSC)
    time_list.clear()
    for i in range(100):
        sub_m._CAPI_drop_format(_COO)
        tic = time.time()
        sub_m._CAPI_create_format(_COO)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "CSC2COO", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        sub_m._CAPI_drop_format(_CSR)
        tic = time.time()
        sub_m._CAPI_create_format(_CSR)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "COO2CSR", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))

    time_list.clear()
    for i in range(100):
        sub_m._CAPI_drop_format(_CSR)
        tic = time.time()
        sub_m._CAPI_create_format(_DCSR)
        torch.cuda.synchronize()
        toc = time.time()
        time_list.append(toc - tic)
    print("DCSR", "COO2DCSR", num_seeds, num_edges,
          1000 * np.mean(time_list[10:]))
