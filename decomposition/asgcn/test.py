import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
from sampler import *


def benchmark(args, graph, nid, fanouts, n_epoch, features, W, sampler):
    print(
        "####################################################{}".format(
            sampler.__name__
        )
    )
    seedloader = SeedGenerator(
        nid, batch_size=args.batchsize, shuffle=True, drop_last=False
    )

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print(
        "memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB"
    )
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            input_nodes, output_nodes, blocks = sampler(
                graph, seeds, features, W, fanouts, args.use_uva
            )

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append(
            (torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024)
        )

        print(
            "Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(
                epoch, epoch_time[-1], mem_list[-1]
            )
        )

    # use the first epoch to warm up
    print("Average epoch sampling time:", np.mean(epoch_time[1:]))
    print("Average epoch gpu mem peak:", np.mean(mem_list[1:]))
    print("####################################################END")


def train(dataset, args):
    device = args.device
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.long().to(device)
    train_nid = splitted_idx["train"].to("cuda")
    # weight = normalized_laplacian_edata(g)
    weight = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 2)).to("cuda")
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids]
    if args.use_uva and device == "cpu":
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        weight, features = weight.pin_memory(), features.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    m._graph._CAPI_set_data(weight)
    print("Check load successfully:", m._graph._CAPI_metadata(), "\n")

    n_epoch = 6
    if args.dataset == "ogbn-products":
        benchmark(args, m, train_nid, fanouts, n_epoch, features, W, w_o_relabel)
        benchmark(
            args, m, train_nid, fanouts, n_epoch, features, W, w_o_relabel_selection
        )
    elif args.dataset == "ogbn-papers100M":
        benchmark(args, m, train_nid, fanouts, n_epoch, features, W, w_o_relabel)
        benchmark(
            args, m, train_nid, fanouts, n_epoch, features, W, w_o_relabel_selection
        )
        benchmark(args, m, train_nid, fanouts, n_epoch, features, W, w_relabel)
        benchmark(
            args, m, train_nid, fanouts, n_epoch, features, W, w_relabel_selection
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--use-uva",
        type=bool,
        default=False,
        help="Wether to use UVA to sample graph and load feature",
    )
    parser.add_argument(
        "--dataset", default="ogbn-products", help="which dataset to load for training"
    )
    parser.add_argument(
        "--batchsize", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--samples", default="512,512", help="sample size for each layer"
    )
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith("ogbn"):
        dataset = load_graph.load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "livejournal":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/livejournal/livejournal.bin"
        )
    elif args.dataset == "friendster":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/friendster/friendster.bin"
        )
    else:
        raise NotImplementedError
    print(dataset[0])
    train(dataset, args)
