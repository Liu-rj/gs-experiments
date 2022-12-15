import gs
from gs.utils import SeedGenerator
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
from load_graph import *
from model import *
from sampler import *

def load_reddit():
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products",root="/home/ubuntu/.dgl")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g=g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx

def load_100Mpapers():
    data = DglNodePropPredDataset(name="ogbn-papers100M",root="/home/ubuntu/.dgl")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g=g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx

def load_friendster():
    bin_path = "/mzydata/data/friendster_coo_with_feature_large.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    g = g_list[0]
    print("graph loaded")
    train_nid = torch.nonzero(g.ndata["train_mask"].long(), as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"].long(), as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"].long(), as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    g=g.long()
    features = np.random.rand(g.num_nodes(), 64)
    labels = np.random.randint(0, 2, size=g.num_nodes())
    feat = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    n_classes = 2
    g.ndata.clear()
    g.edata.clear()
    print("adding self loop...")
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    splitted_idx = dict()
    splitted_idx['train'] = train_nid
    splitted_idx['valid']=val_nid
    splitted_idx['test']=test_nid
    print(g)
    return g, feat, labels, n_classes, splitted_idx
def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(',')]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g, train_nid, val_nid = g.to(device), train_nid.to(
        device), val_nid.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
    else:
        features, labels = features.to(device), labels.to(device)
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    del g
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    compiled_func = matrix_sampler
    train_seedloader = SeedGenerator(
        train_nid, batch_size=args.batchsize, shuffle=False, drop_last=True)
    val_seedloader = SeedGenerator(
        val_nid, batch_size=args.batchsize, shuffle=False, drop_last=True)
    model = SAGEModel(features.shape[1], 64,
                      n_classes, len(fanouts), 0.0).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epoch = 1

    sample_time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    forward_time_list = []
    backward_time_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(n_epoch):
        epoch_feature_loading = 0
        sample_time = 0
        forward_time = 0
        backward_time = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_train_loss = 0
        total_sample_loss = 0
        torch.cuda.synchronize()
        tic = time.time()
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds = seeds.to('cuda')
            input_nodes, output_nodes, blocks, loss_tuple = compiled_func(
                m, seeds, fanouts, features, model.sample_W, model.sample_W2, model.sample_a, use_uva)
            print("input_nodes:",input_nodes.numel())
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            tic = time.time()
            blocks = [block.to('cuda') for block in blocks]
            if use_uva:
                batch_inputs = gather_pinned_tensor_rows(
                    features, input_nodes)
                batch_labels = gather_pinned_tensor_rows(labels, seeds)
            else:
                batch_inputs = features[input_nodes].to('cuda')
                batch_labels = labels[seeds].to('cuda')
            torch.cuda.synchronize()
            epoch_feature_loading += time.time() - tic

            tic = time.time()
            batch_pred = model(blocks, batch_inputs)
            is_labeled = batch_labels == batch_labels
            batch_labels = batch_labels[is_labeled].long()
            batch_pred = batch_pred[is_labeled]
            torch.cuda.synchronize()
            forward_time += time.time() - tic

            tic = time.time()
            opt.zero_grad()
            train_loss = F.cross_entropy(batch_pred, batch_labels)
            train_loss.backward()
            # Loss for sampling probability function
            # Gradient of intermediate tensor
            chain_grad = model.X1.grad
            # Compute intermediate loss for sampling probability parameters
            sample_loss = sampler_loss(
                loss_tuple, chain_grad.detach(), features, use_uva)
            sample_loss.backward()
            opt.step()
            total_train_loss += train_loss.item()
            total_sample_loss += sample_loss.item()
            torch.cuda.synchronize()
            backward_time += time.time() - tic
            tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            torch.cuda.synchronize()
            tic = time.time()
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds = seeds.to('cuda')
                input_nodes, output_nodes, blocks, _ = compiled_func(
                    m, seeds, fanouts, features, model.sample_W, model.sample_W2, model.sample_a, use_uva)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, seeds)
                else:
                    batch_inputs = features[input_nodes].to('cuda')
                    batch_labels = labels[seeds].to('cuda')
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                tic = time.time()
                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                torch.cuda.synchronize()
                forward_time += time.time() - tic
                val_pred.append(batch_pred)
                val_labels.append(batch_labels)
                tic = time.time()

        acc = compute_acc(torch.cat(val_pred, 0),
                          torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        sample_time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)
        forward_time_list.append(forward_time)
        backward_time_list.append(backward_time)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Forward Time {:.4f} s | Backward Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, acc, epoch_time[-1], forward_time_list[-1], backward_time_list[-1], sample_time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[0:]))
    print('Average epoch forward time:', np.mean(forward_time_list[0:]))
    print('Average epoch backward time:', np.mean(backward_time_list[0:]))
    print('Average epoch sampling time:', np.mean(sample_time_list[0:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[0:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[0:]))

    # print('Testing...')
    # acc = layerwise_infer(g, test_nid, model,
    #                       batch_size=4096, feat=features, label=labels)
    # print("Test Accuracy {:.4f}".format(acc.item()))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_seed(1)
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--samples", default='10,10',
                        help="sample size for each layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    args = parser.parse_args()
    print(args)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogbn_products()
    elif args.dataset == 'papers100m':
        dataset = load_100Mpapers()
    print(dataset[0])
    train(dataset, args)
