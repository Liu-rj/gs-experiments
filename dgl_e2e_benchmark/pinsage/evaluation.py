import numpy as np
import torch
import pickle
import dgl
import argparse
import time


def prec(recommendations, val_g):
    n_users = val_g.num_nodes('_U')
    K = recommendations.shape[1]
    user_idx = torch.repeat_interleave(val_g.nodes('_U'), K)
    item_idx = recommendations.flatten()
    torch.cuda.synchronize()
    start = time.time()
    relevance = val_g.has_edges_between(
        user_idx, item_idx, '_E').reshape((n_users, K))
    torch.cuda.synchronize()
    end = time.time()
    hit = torch.any(relevance, dim=1)
    hit = hit.sum().div(len(hit))
    return hit.item(), end - start


class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        torch.cuda.synchronize()
        start = time.time()
        graph_slice = full_graph.edge_type_subgraph(
            [self.user_to_item_etype]).to('cpu')
        n_users = full_graph.num_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(
            graph_slice, 1, self.timestamp, edge_dir='out')
        user, latest_items = latest_interactions.all_edges(
            form='uv', order='srcdst')
        latest_items = latest_items.to(device=h_item.device)
        torch.cuda.synchronize()
        end = time.time()
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch]
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(
                    u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations, end - start


def evaluate_nn(dataset, val_g, h_item, k, batch_size):
    g = dataset['train-graph']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)

    recommendations, sampling_time_1 = rec_engine.recommend(g, k, None, h_item)
    res, sampling_time_2 = prec(recommendations, val_g)
    return res, sampling_time_1 + sampling_time_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
