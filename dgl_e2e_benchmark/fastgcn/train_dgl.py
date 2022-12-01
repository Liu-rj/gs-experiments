import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader
import tqdm
from model import *
from sampler import *
from load_graph import *
import argparse
from dgl.utils import gather_pinned_tensor_rows
import time


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args['device']
    use_uva = args['use_uva']

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g, train_nid, val_nid = g.to(device), train_nid.to(
        device), val_nid.to(device)
    probs = g.out_degrees().float()
    g = g.formats('csc')
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        probs = probs.pin_memory()
    else:
        features, labels = features.to(device), labels.to(device)
        probs = probs.to(device)

    num_nodes = args['num_nodes']
    sampler = FastGCNSampler(num_nodes, replace=False, probs=probs)
    print(g.formats())
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'], use_uva=use_uva)
    val_dataloader = DataLoader(
        g,
        val_nid,
        sampler,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=args['num_workers'], use_uva=use_uva)

    model = Model(features.shape[1],
                  args['hidden_dim'], n_classes, len(num_nodes)).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')

    for epoch in range(args['num_epochs']):
        epoch_feature_loading = 0
        sample_time = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        torch.cuda.synchronize()
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(tqdm.tqdm(train_dataloader)):
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

            batch_pred = model(blocks, batch_inputs)
            is_labeled = batch_labels == batch_labels
            batch_labels = batch_labels[is_labeled].long()
            batch_pred = batch_pred[is_labeled]
            loss = F.cross_entropy(batch_pred, batch_labels)
            # acc = compute_acc(batch_pred, batch_labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # train_dataloader.set_postfix({'loss': '%.06f' % loss.item(),
            #                 'acc': '%.03f' % acc.item()})
            torch.cuda.synchronize()
            tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            torch.cuda.synchronize()
            tic = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(tqdm.tqdm(val_dataloader)):
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

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                val_pred.append(batch_pred)
                val_labels.append(batch_labels)
                torch.cuda.synchronize()
                tic = time.time()

        acc = compute_acc(torch.cat(val_pred, 0),
                          torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, acc, epoch_time[-1], time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))


if __name__ == '__main__':
    config = {
        'num_epochs': 5,
        'hidden_dim': 64}

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--samples", default='2000,2000',
                        help="sample size for each layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    args = parser.parse_args()
    config['device'] = args.device
    config['use_uva'] = args.use_uva
    config['dataset'] = args.dataset
    config['batch_size'] = args.batchsize
    config['num_nodes'] = [int(x.strip()) for x in args.samples.split(',')]
    config['num_workers'] = args.num_workers
    print(config)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M')
    print(dataset[0])
    train(dataset, config)
