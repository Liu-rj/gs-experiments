import rmm
import cugraph
import cudf
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm


def time_graphsage(graph, seeds, batchsize, fanout, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            result, duration = cugraph.uniform_neighbor_sample(
                graph, sub_slicing, fanout, with_replacement=False
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


def time_deepwalk(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes, duration = cugraph.random_walks(
                graph,
                random_walks_type="uniform",
                start_vertices=sub_slicing,
                max_depth=walk_length,
                legacy_result_type=True,
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


def time_node2vec(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        epoch_time = 0
        for j in tqdm(range(batchnum)):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes, duration = cugraph.node2vec(
                graph,
                start_vertices=sub_slicing,
                max_depth=walk_length,
                compress_result=True,
                p=2.0,
                q=0.5,
            )
            epoch_time += duration
        time_list.append(epoch_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} s".format(
                len(seeds), batchnum, time_list[-1]
            ),
            flush=True,
        )
    print("average epoch run time:", np.mean(time_list[1:]), "s", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="livejournal",
    choices=["papers100m", "friendster", "livejournal"],
    help="which dataset to load for training",
)
args = parser.parse_args()

rmm.reinitialize(managed_memory=True)
assert rmm.is_initialized()
# cudf.set_allocator("managed")

start_ts = time.time()
if args.dataset == "papers100m":
    print("load papers100m", flush=True)
    cdf = cudf.read_csv(
        "/home/ubuntu/data/papers100m/papers100m.csv",
        skiprows=1,
        names=["src", "dst"],
        dtype=["int64", "int64"],
    )
    train_id = torch.load("/home/ubuntu/data/papers100m/papers100m_train_id.pt")
    train_id = train_id.cpu().numpy()
    index = np.random.permutation(train_id.shape[0])
    permuted_nid = train_id[index]
elif args.dataset == "friendster":
    print("load friendster", flush=True)
    cdf = cudf.read_csv(
        "/home/ubuntu/data/friendster/friendster.csv",
        skiprows=1,
        names=["src", "dst"],
        dtype=["int64", "int64"],
    )
    train_id = torch.load("/home/ubuntu/data/friendster/friendster_trainid.pt")
    train_id = train_id.cpu().numpy()
    index = np.random.permutation(train_id.shape[0])[: int(train_id.shape[0] / 10)]
    permuted_nid = train_id[index]
elif args.dataset == "livejournal":
    print("load livejournal", flush=True)
    cdf = cudf.read_csv(
        "/home/ubuntu/data/livejournal/livejournal.csv",
        skiprows=1,
        names=["src", "dst"],
        dtype=["int64", "int64"],
    )
    train_id = torch.load("/home/ubuntu/data/livejournal/livejournal_trainid.pt")
    train_id = train_id.cpu().numpy()
    index = np.random.permutation(train_id.shape[0])
    permuted_nid = train_id[index]
else:
    raise NotImplementedError
print("csv and train_id loaded", flush=True)

G = cugraph.MultiGraph(directed=True)
G.from_cudf_edgelist(cdf, source="src", destination="dst", renumber=False)
print("graph created", flush=True)
end_ts = time.time()
print("Total Time on plain UVM  approach: " + str(end_ts - start_ts) + " s", flush=True)

permuted_nid = cudf.Series(permuted_nid)

print("Timing graphsage", flush=True)
batchsize = 512
fanout = [25, 10]
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
time_graphsage(G, permuted_nid, batchsize, fanout, batchnum)


print("Timing deepmwalk", flush=True)
batchsize = 1024
walk_len = 80
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
time_deepwalk(G, permuted_nid, batchsize, walk_len, batchnum)


print("Timing node2vec", flush=True)
batchsize = 1024
walk_len = 80
batchnum = int((permuted_nid.shape[0] + batchsize - 1) / batchsize)
time_node2vec(G, permuted_nid, batchsize, walk_len, batchnum)
