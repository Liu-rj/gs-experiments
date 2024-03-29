import torch
import dgl
from gs.utils import load_graph
import time
import numpy as np
import gs
from gs.utils import SeedGenerator
import sys 
sys.path.append("..") 
from ogb.nodeproppred import DglNodePropPredDataset
import argparse
from dgl.dataloading import DataLoader, NeighborSampler
import tqdm
import scipy.sparse as sp

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products",root="/home/ubuntu/data")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g=g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print("products")
    return g, feat, labels, n_classes, splitted_idx

def load_100Mpapers():
    train_id = torch.load("/home/ubuntu/data/papers100m_train_id.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/data/ogbn-papers100M_adj.npz")
    
    g = dgl.from_scipy(coo_matrix)
    print("before:",g)
    # g = g.formats("csc")
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g=g.long()
    return g, None, None, None, splitted_idx

def load_livejournal():
    train_id = torch.load("/home/ubuntu/data/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/data/livejournal/livejournal_adj.npz")

    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g=g.long()
    return g, None, None, None, splitted_idx

def load_friendster():
    train_id = torch.load("/home/ubuntu/data/friendster_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    # bin_path = "/home/ubuntu/data/friendster/friendster.bin"
    # g_list, _ = dgl.load_graphs(bin_path)
    # g = g_list[0]
    # print("graph loaded")
    # train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    # test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    # val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]

    # features = np.random.rand(g.num_nodes(), 128)
    # labels = np.random.randint(0, 3, size=g.num_nodes())
    # feat = torch.tensor(features, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # n_classes = 3

    coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
    csc_matrix = coo_matrix.tocsc()
    g = dgl.from_scipy(csc_matrix)
    # g = g.formats("csc")
    # g=g.long()
    return g, None,None,None,splitted_idx

class node2vecSampler(object):
    def __init__(self,walk_length, p=0.5,q=2,):
        super().__init__()
        self.walk_length = walk_length
        self.p=p
        self.q=q
    def sample(self, g, seeds,exclude_eids=None):
        torch.cuda.nvtx.range_push('dgl random walk')
        traces = dgl.sampling.node2vec_random_walk(g, seeds, self.p,self.q,self.walk_length)
        return traces




def benchmark_w_o_relabel(args, graph, nid):
    print('####################################################DGL deepwalk')
    sampler = node2vecSampler(args.walk_length)
    print("train id size:",len(nid))
    # seedloader = SeedGenerator(
    #     nid, batch_size=args.batchsize, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(graph, nid, sampler,batch_size=args.batchsize, use_prefetch_thread=False,
    shuffle=True,drop_last=False, num_workers=args.num_workers,device='cpu',use_uva=False)
    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        # with train_dataloader.enable_cpu_affinity():
        for it, seeds in enumerate(tqdm.tqdm(train_dataloader)):
            pass
            # traces = sampler.sample(graph, seeds)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
            .format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print('Average epoch sampling time:', np.mean(epoch_time[1:])*1000," ms")
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:])," GB")
    print('####################################################END')

    # sample_list = []
    # static_memory = torch.cuda.memory_allocated()
    # print('memory allocated before training:',
    #       static_memory / (1024 * 1024 * 1024), 'GB')
    # tic = time.time()
    # with tqdm.tqdm(train_dataloader) as tq:
    #     for step, walks in enumerate(tq):
    #         if step > 50:
    #             break
    #         torch.cuda.synchronize()
    #         sampling_time=time.time()-tic
    #         sample_list.append(sampling_time)
    #         # print(sampling_time)
    #         sampling_time = 0
    #         torch.cuda.synchronize()
    #         tic=time.time()
            
    # print('Average epoch sampling time:', np.mean(sample_list[2:]))
def load(dataset,args):
    device = args.device
    use_uva = args.use_uva
    g, features, labels, n_classes, splitted_idx = dataset
    sample_list = []
    static_memory = torch.cuda.memory_allocated()
    train_nid = splitted_idx['train']
    if args.data_type == 'int':
        g = g.int()
        train_nid = train_nid.int()
    else:
        g = g.long()
        train_nid = train_nid.long()
    # g = g.to(device)
    # train_nid = train_nid.to('cuda')
    # csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    if use_uva and device == 'cpu':
        g.pin_memory_()
         # csc_indptr = csc_indptr.pin_memory()
        # csc_indices = csc_indices.pin_memory()
    benchmark_w_o_relabel(args, g, train_nid)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='products', choices=['reddit', 'products', 'papers100m','friendster','livejournal'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=128,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=3,
                        help="numbers of epoch in training")
    parser.add_argument("--sample-mode", default='ad-hoc', choices=['ad-hoc', 'fine-grained','matrix-fused','matrix-nonfused'],
                        help="sample mode")
    parser.add_argument("--data-type", default='long', choices=['int', 'long'],
                        help="data type")
    parser.add_argument("--walk-length", type=int, default=80,
                        help="random walk walk length")
    args = parser.parse_args()
    print('Loading data')
    if args.dataset == 'products':
        dataset = load_ogbn_products()
    elif args.dataset == 'papers100m':
        dataset = load_100Mpapers()
    elif args.dataset == 'friendster':
        dataset = load_friendster()
    elif args.dataset == 'livejournal':
        dataset = load_livejournal()
    print(dataset[0])


# bench('DGL random walk', dgl_sampler, g, 4, iters=10, node_idx=nodes)
# bench('Matrix random walk Non-fused', matrix_sampler_nonfused, matrix,
#       4, iters=10, node_idx=nodes)
    load(dataset,args)
