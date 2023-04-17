#!/usr/bin/env python
# coding: utf-8


from utils import *
from tqdm import tqdm
import argparse
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(
    description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='ogbn-products',
                    help='Dataset name')
parser.add_argument('--epoch_num', type=int, default=1,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')


args = parser.parse_args()


def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes, :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        print(p, p.shape)
        print(np.isnan(p).sum())
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p=p, replace=False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.
        adj = U[:, after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes


def prepare_data(pool, sampler, process_num, train_nodes, samp_num_list, num_nodes, lap_matrix, depth, batch_size):
    jobs = []
    for i in range(process_num):
        end = len(train_nodes) if i == process_num - \
            1 else (i + 1) * batch_size
        batch_nodes = train_nodes[i * batch_size:end]
        p = pool.apply_async(sampler, args=(np.random.randint(
            2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    return jobs


adj, train_nodes = load_npz(args.dataset)
print('####################################################{}'.format(args.dataset))
num_nodes = adj.shape[0]

adj = adj.astype(np.float32)
adj_matrix = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

idx = torch.randperm(len(train_nodes))
train_nodes = train_nodes[idx]
batch_num = int((len(train_nodes) + args.batch_size - 1) / args.batch_size)
iter_num = int((batch_num + args.pool_num - 1) / args.pool_num)
samp_num_list = np.array(
    [args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

epoch_time = []
for epoch in np.arange(args.epoch_num):
    start = time.time()

    pool = mp.Pool(args.pool_num)
    for it in tqdm(range(iter_num)):
        start = it * args.pool_num * args.batch_size
        end = len(train_nodes) if it == iter_num - \
            1 else (it + 1) * args.pool_num * args.batch_size
        jobs = prepare_data(pool, ladies_sampler, args.pool_num, train_nodes[start:end],
                            samp_num_list, num_nodes, lap_matrix, args.n_layers, args.batch_size)
        train_data = [job.get() for job in jobs]
    pool.close()
    pool.join()

    epoch_time.append(time.time() - start)
    print("Epoch {:05d} | Epoch Sample Time {:.4f} s".format(
        epoch, epoch_time[-1]))

print('Average epoch sampling time:', np.mean(epoch_time))
print('####################################################END')
