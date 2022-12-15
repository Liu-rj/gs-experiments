
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import pin_memory_inplace
from model import *
from sampler import *
import time
import argparse
import os
from ctypes import *
from ctypes.util import *
import numpy as np
from load_graph import *
import torch.nn.functional as F

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


def train_dgl(dataset, config):
    use_uva = config['use_uva']
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']

    if use_uva:
        features, labels = features.pin_memory(), labels.pin_memory()
    elif config['device'] == 'cuda':
        g, train_nid, val_nid = g.to('cuda'), train_nid.to(
            'cuda'), val_nid.to('cuda')
        features = features.to('cuda')
        labels = labels.to('cuda')
    else:
        features = features.to('cuda')
        labels = labels.to('cuda')
    batch_size = config['batch_size']
    fanouts = [10, 10]
    model = SAGEModel(
        features.shape[1], 64, n_classes, len(fanouts), 0.0).to('cuda')
    sampler = DGLNeighborSampler(
        fanouts, model.sample_W, model.sample_W2, model.sample_a, use_uva, features=features)
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=batch_size,
                                  shuffle=False,  drop_last=True, num_workers=config['num_workers'], device='cuda')
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=batch_size, shuffle=False,
                                drop_last=True, num_workers=config['num_workers'], device='cuda')

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sample_list = []
    epoch_list = []
    mem_list = []
    feature_loading_list = []
    forward_time_list = []
    backward_time_list = []
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(config['num_epoch']):
        epoch_feature_loading = 0
        sampling_time = 0
        forward_time = 0
        backward_time = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        model.train()
        tic = time.time()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, blocks) in enumerate(tq):
              #  print("step ",step,": ,seeds:",output_nodes)
                output_nodes = output_nodes.to('cuda')
                torch.cuda.synchronize()
                sampling_time += time.time() - tic

                tic = time.time()
                if use_uva:
                    x = gather_pinned_tensor_rows(features, input_nodes)
                    y = gather_pinned_tensor_rows(labels, output_nodes)
                else:
                    x = features[input_nodes].to('cuda')
                    y = labels[output_nodes].to('cuda')
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                tic = time.time()
                y_hat = model(blocks, x)
                is_labeled = y == y
                y = y[is_labeled].long()
                y_hat = y_hat[is_labeled]
                torch.cuda.synchronize()
                forward_time += time.time() - tic

                tic = time.time()
                opt.zero_grad()
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                chain_grad = model.X1.grad
                # Compute intermediate loss for sampling probability parameters
                sample_loss = sampler_loss(
                    sampler.ret_loss_tuple, chain_grad.detach(), features, use_uva)
                sample_loss.backward()
                opt.step()
                torch.cuda.synchronize()
                backward_time += time.time() - tic
                acc = compute_acc(y_hat, y)
                tq.set_postfix({'loss': '%.06f' % loss.item(),
                                'acc': '%.03f' % acc.item()})
                tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        tic = time.time()
        with torch.no_grad():
            with tqdm.tqdm(val_dataloader) as tq:
                for it, (input_nodes, output_nodes, blocks) in enumerate(tq):
                    output_nodes = output_nodes.to('cuda')
                    torch.cuda.synchronize()
                    sampling_time += time.time() - tic

                    tic = time.time()
                    if use_uva:
                        x = gather_pinned_tensor_rows(features, input_nodes)
                        y = gather_pinned_tensor_rows(labels, output_nodes)
                    else:
                        x = features[input_nodes].to('cuda')
                        y = labels[output_nodes].to('cuda')
                    torch.cuda.synchronize()
                    epoch_feature_loading += time.time() - tic

                    tic = time.time()
                    y_hat = model(blocks, x)
                    is_labeled = y == y
                    y = y[is_labeled].long()
                    y_hat = y_hat[is_labeled]
                    torch.cuda.synchronize()
                    forward_time += time.time() - tic
                    val_pred.append(y_hat)
                    val_labels.append(y)
                    tic = time.time()
           # acc = compute_acc(val_pred,val_labels)

        torch.cuda.synchronize()
        epoch_list.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        sample_list.append(sampling_time)
        feature_loading_list.append(epoch_feature_loading)
        forward_time_list.append(forward_time)
        backward_time_list.append(backward_time)

        print("Epoch {:05d} | E2E Time {:.4f} s | Forward Time {:.4f} s | Backward Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_list[-1], forward_time_list[-1], backward_time_list[-1], sample_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[1:]))
    print('Average epoch sampling time:', np.mean(sample_list[1:]))
    print('Average epoch end2end time:', np.mean(epoch_list[1:]))
    print('Average epoch forward time:', np.mean(forward_time_list[1:]))
    print('Average epoch backward time:', np.mean(backward_time_list[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    config = {}
    setup_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=5,
                        help="numbers of epoch in training")
    args = parser.parse_args()
    config['device'] = args.device
    config['use_uva'] = args.use_uva
    config['dataset'] = args.dataset
    config['batch_size'] = args.batchsize
    config['num_workers'] = args.num_workers
    config['num_epoch'] = args.num_epoch
    print(config)
    print('Loading data')
    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogbn_products()
    elif args.dataset == 'papers100m':
        dataset = load_100Mpapers()
    print(dataset[0])
    if args.device != 'cuda':
        config['feat_device'] = 'cpu'
    else:
        config['feat_device'] = 'cuda'
    train_dgl(dataset, config)
