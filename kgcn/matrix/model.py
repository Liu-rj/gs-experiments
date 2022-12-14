import torch
from aggregator import Aggregator
import gs
import time
import numpy as np


class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg: gs.Matrix, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(
            self.batch_size, self.dim, args.aggregator)

        # self._gen_adj()
        time_list = []
        for i in range(10):
            torch.cuda.synchronize()
            start = time.time()
            self._gen_adj()
            torch.cuda.synchronize()
            time_list.append(time.time() - start)
        print('full graph average sampling time:', np.mean(time_list[3:]))
        self.epoch_sample_time = np.mean(time_list[3:])

        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        # torch.cuda.nvtx.range_push('full graph sampling')
        # torch.cuda.nvtx.range_push('column sampling')
        sampled_adj_matrix = self.kg.columnwise_sampling(self.n_neighbor, True)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('coo rows')
        self.adj_ent = sampled_adj_matrix.row_indices().view(
            (self.num_ent, self.n_neighbor))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('get data')
        self.adj_rel = sampled_adj_matrix.get_data().view((self.num_ent, self.n_neighbor))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()

    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))

        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim=1)

        entities, relations = self._get_neighbors(v)

        item_embeddings = self._aggregate(user_embeddings, entities, relations)

        scores = (user_embeddings * item_embeddings).sum(dim=1)

        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        torch.cuda.synchronize()
        start = time.time()
        
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = self.adj_ent[entities[h]].view(
                (self.batch_size, -1))
            neighbor_relations = self.adj_rel[entities[h]].view(
                (self.batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        
        torch.cuda.synchronize()
        self.epoch_sample_time += time.time() - start

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))
