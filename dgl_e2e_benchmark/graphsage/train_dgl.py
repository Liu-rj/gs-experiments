
from dgl.dataloading import DataLoader, NeighborSampler
from gs.utils import SeedGenerator,ConvModel
from dgl.utils import pin_memory_inplace
from model import *
from sampler import *
import time
import argparse
import os
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
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
    data = DglNodePropPredDataset(name="ogbn-products", root="/home/ubuntu/.dgl")
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

def train_matrix(dataset, config):
    device = config['device']
    use_uva = config['use_uva']
    fanouts = config['fanouts']
    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    del g
    del edge_ids
    if use_uva:
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
    elif config['device'] == 'cuda':
        csc_indptr = csc_indptr.to('cuda')
        csc_indices = csc_indices.to('cuda')
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    print("matrix load successfully!")
    #print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    if config['sample_mode'] == 'matrix-fused':
        compiled_func = matrix_sampler_fused
    else:
        compiled_func = matrix_sampler_nonfused
    train_seedloader = SeedGenerator(
        train_nid, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_nid, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    model = GraphSAGE_DGL(
        features.shape[1], 64, n_classes, len(fanouts), use_uva).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sample_time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(config['num_epoch']):
        epoch_feature_loading = 0
        sample_time = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        model.train()
        torch.cuda.synchronize()
        tic = time.time()
        with tqdm.tqdm(train_seedloader) as tq:
            for it, seeds in enumerate(tq):
                input_nodes, output_nodes, blocks = compiled_func(m, seeds, fanouts)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, seeds)
                else:
                    batch_inputs = features[input_nodes]
                    batch_labels = labels[seeds]

                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                batch_pred = model(blocks, batch_inputs)
                # is_labeled = batch_labels == batch_labels
                # batch_labels = batch_labels[is_labeled].long()
                # batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = compute_acc(batch_pred, batch_labels)
                tq.set_postfix({'loss': '%.06f' % loss.item(),'acc': '%.03f' % acc.item()})
                torch.cuda.synchronize()
                tic = time.time()

        # model.eval()
        # val_pred = []
        # val_labels = []
        # with torch.no_grad():
        #     torch.cuda.synchronize()
        #     tic = time.time()
        #     for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
        #         seeds = seeds.to('cuda')
        #         input_nodes, output_nodes, blocks = compiled_func(m, seeds, fanouts)
        #         torch.cuda.synchronize()
        #         sample_time += time.time() - tic

        #         tic = time.time()
        #         blocks = [block.to('cuda') for block in blocks]
        #         if use_uva:
        #             batch_inputs = gather_pinned_tensor_rows(
        #                 features, input_nodes)
        #             batch_labels = gather_pinned_tensor_rows(labels, seeds)
        #         else:
        #             batch_inputs = features[input_nodes].to('cuda')
        #             batch_labels = labels[seeds].to('cuda')
        #         torch.cuda.synchronize()
        #         epoch_feature_loading += time.time() - tic

        #         batch_pred = model(blocks, batch_inputs)
        #         # is_labeled = batch_labels == batch_labels
        #         # batch_labels = batch_labels[is_labeled].long()
        #         # batch_pred = batch_pred[is_labeled]

        #         val_pred.append(batch_pred)
        #         val_labels.append(batch_labels)
        #         tic = time.time()

        #acc = compute_acc(torch.cat(val_pred, 0),torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        sample_time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch,  epoch_time[-1], sample_time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[1:]))
    print('Average epoch sampling time:', np.mean(sample_time_list[1:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))

def train_dgl(dataset, config):
    device = config['device']
    use_uva = config['use_uva']
    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    batch_size = config['batch_size']
    fanouts = config['fanouts']


    sampler = None
    if config['sample_mode'] == 'ad-hoc':
        sampler = DGLNeighborSampler(fanouts)
    else:
        sampler = DGLNeighborSampler_finegrained(fanouts)
    print("graph device:",g.device)
    print("train nid device:",train_nid.device)
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=config['num_workers'], device='cuda',use_uva=use_uva)
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=batch_size, shuffle=True,drop_last=False, num_workers=config['num_workers'], device='cuda',use_uva=use_uva)
    model = GraphSAGE_DGL(features.shape[1], 64, n_classes, len(fanouts), use_uva).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    sample_list = []
    epoch_list = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(config['num_epoch']):
        epoch_feature_loading = 0
        sampling_time = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        model.train()
        torch.cuda.synchronize()
        tic = time.time()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, blocks) in enumerate(tq):
                torch.cuda.synchronize()
                sampling_time += time.time()-tic

                temp = time.time()
                if use_uva:
                    x = gather_pinned_tensor_rows(
                        features, input_nodes.to('cuda'))
                    y = gather_pinned_tensor_rows(
                        labels, output_nodes.to('cuda'))
                else:
                    x = features[input_nodes]
                    y = labels[output_nodes]

                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - temp

                y_hat = model(blocks, x)
                # is_labeled = y == y
                # y = y[is_labeled].long()
                # y_hat = y_hat[is_labeled]
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = compute_acc(y_hat, y)
                tq.set_postfix({'loss': '%.06f' % loss.item(),'acc': '%.03f' % acc.item()})
                torch.cuda.synchronize()
                tic = time.time()

        # model.eval()
        # val_pred = []
        # val_labels = []
        # tic = time.time()
        # with torch.no_grad():
        #     with tqdm.tqdm(val_dataloader) as tq:
        #         for it, (input_nodes, output_nodes, blocks) in enumerate(tq):
        #             torch.cuda.synchronize()
        #             sampling_time += time.time() - tic

        #             temp = time.time()
        #             if use_uva:
        #                 x = gather_pinned_tensor_rows(
        #                     features, input_nodes.to('cuda'))
        #                 y = gather_pinned_tensor_rows(
        #                     labels, output_nodes.to('cuda'))
        #             else:
        #                 x = features[input_nodes]
        #                 y = labels[input_nodes]
        #             torch.cuda.synchronize()
        #             epoch_feature_loading += time.time() - temp

        #             y_pred = model(blocks, x)
        #             val_pred.append(y_pred)
        #             val_labels.append(y)
        #             tic = time.time()
        #acc = compute_acc(val_pred,val_labels)

        torch.cuda.synchronize()
        epoch_list.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        sample_list.append(sampling_time)
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_list[-1], sample_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[1:]))
    print('Average epoch sampling time:', np.mean(sample_list[1:]))
    print('Average epoch end2end time:', np.mean(epoch_list[1:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:]))


if __name__ == '__main__':
    config = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m','friendster'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1024,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=4,
                        help="numbers of epoch in training")
    parser.add_argument("--sample-mode", default='ad-hoc', choices=['ad-hoc', 'fine-grained','matrix-fused','matrix-nonfused'],
                        help="sample mode")
    parser.add_argument("--samples", default='10,10,10',help="sample size for each layer")
    args = parser.parse_args()
    config['device'] = args.device
    config['use_uva'] = args.use_uva
    config['dataset'] = args.dataset
    config['batch_size'] = args.batchsize
    config['num_workers'] = args.num_workers
    config['num_epoch'] = args.num_epoch
    config['sample_mode'] = args.sample_mode
    config['fanouts'] = [int(x.strip()) for x in args.samples.split(',')]
   
    print('Loading data')
    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogbn_products()
    elif args.dataset == 'papers100m':
        dataset = load_100Mpapers()

    elif args.dataset == 'friendster':
        dataset = load_friendster()

    device = config['device']
    use_uva = config['use_uva']
    g, features, labels, n_classes, splitted_idx = dataset
    print(features.shape)
    g = g.formats('csc')
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    if use_uva:
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
        features, labels = features.pin_memory(), labels.pin_memory()
    elif config['device'] == 'cuda':
        g, train_nid, val_nid = g.to('cuda'), train_nid.to('cuda'), val_nid.to('cuda')
        features = features.to('cuda')
        labels = labels.to('cuda')
    else: 
        features = features.to('cuda')
        labels = labels.to('cuda')
    splitted_idx['train'], splitted_idx['valid'], splitted_idx['test'] = train_nid, val_nid, test_nid 
    dataset = (g, features, labels, n_classes, splitted_idx)
    print(dataset[0])

    
    if config['sample_mode'] == 'ad-hoc' or config['sample_mode'] == 'fine-grained':
        print(config)
        train_dgl(dataset,config)
    else:
        print(config)
        train_matrix(dataset,config)
    # print(config)
    # train_matrix(dataset,config)
    # config['sample_mode'] = 'matrix-fused'
    # print(config)
    # train_dgl(dataset,config)




