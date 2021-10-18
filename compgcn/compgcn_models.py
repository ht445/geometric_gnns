import torch
from abc import ABC
import torch_geometric
import torch.nn.functional as functional
from torch import LongTensor, FloatTensor


class CompgcnLP(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int, num_bases: int, aggr: str, norm: int, dropout: float):
        super(CompgcnLP, self).__init__()
        self.norm = norm
        self.dropout = dropout

        self.entity_embeds = torch.nn.Parameter(torch.FloatTensor(num_entities, dimension))
        torch.nn.init.kaiming_uniform_(self.entity_embeds, nonlinearity="leaky_relu")  # entity embeddings, size: (num_entities, dimension)

        self.bases = torch.nn.Parameter(torch.FloatTensor(num_bases, dimension))  # base vectors for relations
        torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
        self.coefficients = torch.nn.Parameter(torch.FloatTensor(num_relations, num_bases))  # relation coefficients
        torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

        self.compgcn1 = CompGCN(in_dimension=dimension, out_dimension=dimension, aggr=aggr)
        self.compgcn2 = CompGCN(in_dimension=dimension, out_dimension=dimension, aggr=aggr)

    def encode(self, ent_ids: LongTensor, edge_index: LongTensor, edge_type: LongTensor, y: LongTensor) -> [FloatTensor]:
        x = torch.index_select(input=self.entity_embeds, index=ent_ids, dim=0)  # size: (num_entities_in_the_current_batch, dimension)
        r = torch.matmul(self.coefficients, self.bases)  # size: (num_relations, dimension)

        x, r = self.compgcn1.forward(x=x, r=r, edge_index=edge_index, edge_type=edge_type, y=y)

        # x = functional.leaky_relu(x)
        # r = functional.leaky_relu(r)
        x = functional.dropout(input=x, p=self.dropout, training=self.training)
        r = functional.dropout(input=r, p=self.dropout, training=self.training)

        x, r = self.compgcn2.forward(x=x, r=r, edge_index=edge_index, edge_type=edge_type, y=y)

        x = functional.normalize(input=x, p=2, dim=1)
        return x, r

    def decode(self, x: FloatTensor, r: FloatTensor, triples: LongTensor) -> FloatTensor:
        head_ids = triples[:, 0]
        rel_ids = triples[:, 1]
        tail_ids = triples[:, 2]
        head_embeds = torch.index_select(input=x, index=head_ids, dim=0)  # head entity embeddings, size: (batch_size, dimension)
        rel_embeds = torch.index_select(input=r, index=rel_ids, dim=0)  # relation embeddings, size: (batch_size, dimension)
        tail_embeds = torch.index_select(input=x, index=tail_ids, dim=0)  # tail entity embeddings, size: (batch_size, dimension)
        scores = torch.norm(head_embeds + rel_embeds - tail_embeds, p=self.norm, dim=1)  # size: (batch_size)
        return torch.sigmoid(scores)


class CompGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, aggr: str):
        super(CompGCN, self).__init__(aggr=aggr)

        # weights for original, inverse, and self-loop relations: weights[0, :, :], weights[1, :, :], weights[2, :, :]
        self.weights = torch.nn.Parameter(torch.FloatTensor(3, in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.weights, nonlinearity="leaky_relu")

        # weight for relation embedding projection
        self.relation_weight = torch.nn.Parameter(torch.FloatTensor(in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.relation_weight, nonlinearity="leaky_relu")

    def forward(self, x: FloatTensor, r: FloatTensor, edge_index: LongTensor, edge_type: LongTensor, y: LongTensor) -> [FloatTensor, FloatTensor]:
        # x: input entity embeddings, size: (num_entities, in_dimension);
        # r: input relation embeddings, size: (num_relations, in_dimension);
        # edge_index: graph adjacency matrix in COO format, size: (2, num_edges);
        # edge_type: relation id list, and the order corresponds to edge_index, size: (num_edges);
        # y: relation type list: 0-original, 1-inverse, or 2-self-loop, size: (num_edges);
        updated_x = self.propagate(x=x, r=r, edge_index=edge_index, edge_type=edge_type, y=y)  # propagate messages along edges and compute updated entity embeddings
        updated_r = torch.matmul(r, self.relation_weight)  # size: (num_relations, out_dimension)
        return torch.tanh(updated_x), updated_r  # updated entity embeddings; updated_x size: (num_entities, out_dimension); updated_r size: (num_relations, out_dimension)]

    def message(self, x_j: FloatTensor, r: FloatTensor, edge_type: FloatTensor, y: FloatTensor) -> FloatTensor:
        # x_j: embeddings of source entities, size: (num_edges, in_dimension);
        edge_rel_embeds = torch.index_select(input=r, index=edge_type, dim=0)  # size: (num_edges, in_dimension)
        messages = x_j + edge_rel_embeds  # size: (num_edges, in_dimension)

        edge_weights = torch.index_select(input=self.weights, index=y, dim=0)  # size: (num_edges, in_dimension, out_dimension)
        messages = torch.bmm(messages.unsqueeze(1), edge_weights).squeeze(1)  # (num_edges, out_dimension)
        return messages
