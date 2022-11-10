import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import os
import tqdm
import layers
from sampler import *
import evaluation
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from gs.utils import SeedGenerator
import time


def create_textual_node_features(node_ids, textset, device):
    """
    Creates numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    ndata = {}
    node_ids = node_ids.cpu().numpy()

    for field_name, field in textset.items():
        textlist, vocab, pad_var, batch_first = field

        examples = [textlist[i] for i in node_ids]
        ids_iter = numericalize_tokens_from_iterator(vocab, examples)

        maxsize = max([len(textlist[i]) for i in node_ids])
        ids = next(ids_iter)
        x = torch.asarray([num for num in ids])
        lengths = torch.tensor([len(x)])
        tokens = padding(x, maxsize, pad_var)

        for ids in ids_iter:
            x = torch.asarray([num for num in ids])
            l = torch.tensor([len(x)])
            y = padding(x, maxsize, pad_var)
            tokens = torch.vstack((tokens, y))
            lengths = torch.cat((lengths, l))

        if not batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens.to(device)
        ndata[field_name + '__len'] = lengths.to(device)
    return ndata


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    device = torch.device(args.device)
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix']
    val_g = dgl.bipartite_from_scipy(
        val_matrix, utype='_U', etype='_E', vtype='_V', idtype=torch.long, device=g.device)
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']
    node_ids = torch.arange(g.num_nodes(item_ntype), device=g.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(
        g.num_nodes(user_ntype), device=g.device)
    g.nodes[item_ntype].data['id'] = torch.arange(
        g.num_nodes(item_ntype), device=g.device)

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(g.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"])
        textset[key] = (textlist, vocab2, vocab2.get_stoi()
                        ['<pad>'], batch_first)
    features = create_textual_node_features(node_ids, textset, device)

    # Sampler
    batch_sampler = ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    dataloader_test = SeedGenerator(node_ids, batch_size=args.batch_size)
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(g, item_ntype, textset,
                         args.hidden_dims, args.num_layers).to('cuda')
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')

    # For each batch of head-tail-negative triplets...
    for epoch in range(args.num_epochs):
        epoch_feature_loading = 0
        sample_time = 0
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        torch.cuda.synchronize()
        tic = time.time()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            tic = time.time()
            pos_graph, neg_graph = pos_graph.to('cuda'), neg_graph.to('cuda')
            blocks = [block.to('cuda') for block in blocks]
            assign_features_to_blocks(blocks, g, features, item_ntype)
            torch.cuda.synchronize()
            epoch_feature_loading += time.time() - tic

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            tic = time.time()

        # Evaluate
        model.eval()
        with torch.no_grad():
            h_item_batches = []
            torch.cuda.synchronize()
            tic = time.time()
            for batch_id, seeds in enumerate(tqdm.tqdm(dataloader_test)):
                blocks = collator.collate_test(seeds)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                assign_features_to_blocks(
                    blocks, g, features, item_ntype)
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic
                h_item_batches.append(model.get_repr(blocks))
                torch.cuda.synchronize()
                tic = time.time()
            h_item = torch.cat(h_item_batches, 0)

            # hit, sampling_time = evaluation.evaluate_nn(
            #     dataset, val_g, h_item, args.k, args.batch_size)
            # print(hit)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_time[-1], time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=10)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=10)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, 'data.pkl')
    with open(data_info_path, 'rb') as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, 'train_g.bin')
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset['train-graph'] = g_list[0].to(args.device)
    train(dataset, args)
