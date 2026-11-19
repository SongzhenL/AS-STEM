from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
import numpy as np
from torch_geometric.nn import GATConv

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size] # 500, 4
                        q = self.get_queries(these_queries) # 500, 400
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

class MLP(nn.Module):#jia def MLP模块，用于处理关系和实体的交互
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
            #jia 修改，加入MLP

class AS(TKBCModel):
    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, no_time_emb=False, init_size: float = 1e-2,
                 hidden_dim: int = 100, gat_hidden_dim: int = 64, gat_heads: int = 8):
        super(AS, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

        # Define MLP
        self.mlp = MLP(input_dim=4 * rank, hidden_dim=hidden_dim, output_dim=2 * rank)

        # Define GAT
        # self.gat = GATConv(in_channels=2 * rank, out_channels=gat_hidden_dim, heads=gat_heads, dropout=0.6)
        self.gat = GATConv(in_channels=2 * rank, out_channels=2 * rank, heads=1, dropout=0.6)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], \
             rel[1] * time[1] + time_phase[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # Concatenate lhs and full_rel for MLP
        combined = torch.cat([lhs[0], lhs[1], full_rel[0], full_rel[1]], dim=1)
        combined = self.mlp(combined)

        # Split the output of MLP back into real and imaginary parts
        combined = combined[:, :self.rank], combined[:, self.rank:]

        # Apply GAT
        # Assuming edge_index is provided (need to construct it from data)
        edge_index = self.get_edge_index()  # This function needs to be defined
        combined = torch.cat(combined, dim=1)
        combined = self.gat(combined, edge_index)
        combined = combined[:, :self.rank], combined[:, self.rank:]

        return torch.sum(
            (combined[0] * rhs[0] - combined[1] * rhs[1]) +
            (combined[1] * rhs[0] + combined[0] * rhs[1]), 1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        time_phase = torch.abs(self.embeddings[3](x[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], rel[
            1] * time[1] + time_phase[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        # Concatenate lhs and full_rel for MLP
        combined = torch.cat([lhs[0], lhs[1], full_rel[0], full_rel[1]], dim=1)
        combined = self.mlp(combined)

        # Split the output of MLP back into real and imaginary parts
        combined = combined[:, :self.rank], combined[:, self.rank:]

        # Apply GAT
        edge_index = self.get_edge_index()  # This function needs to be defined
        combined = torch.cat(combined, dim=1)
        combined = self.gat(combined, edge_index)
        combined = combined[:, :self.rank], combined[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        # Ensure the shapes match
        assert combined[0].shape[1] == right[0].shape[1], "Shapes do not match"

        return (
                (combined[0] @ right[0].t() - combined[1] @ right[1].t()) +
                (combined[1] @ right[0].t() + combined[0] @ right[1].t())), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight, self.embeddings[3].weight[
                                                                                              :-1] if self.no_time_emb else \
        self.embeddings[3].weight
                                                         :-1] if self.no_time_emb else self.embeddings[3].weight

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / ( 1 / self.pi), rel[:, self.rank:] / ( 1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        time_phase = torch.abs(self.embeddings[3](queries[:, 3]))
        time_phase = torch.sin(time_phase[:, :self.rank]), torch.sin(time_phase[:, self.rank:])

        rt = rel[0] * time[0] + time_phase[0], rel[1] * time[0] + time_phase[0], rel[0] * time[1] + time_phase[1], rel[1] * time[1] + time_phase[1]
        
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return torch.cat([
           lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
           lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)
    def get_edge_index(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).cuda()
        return edge_index
