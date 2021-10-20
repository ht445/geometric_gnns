import torch
from abc import ABC
import torch_scatter
import torch_geometric
import torch.nn.functional as functional
from torch import Tensor, LongTensor, FloatTensor


class RgcnLP(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dimension: int, num_bases: int, aggr: str,
                 dropout: float):
        super(RgcnLP, self).__init__()
        self.dropout = dropout  # dropout rate

        # entity embeddings
        self.ent_embeds = torch.nn.Parameter(torch.FloatTensor(num_entities, dimension))  # entity embeddings, size: (num_entities, dimension)
        torch.nn.init.xavier_normal_(self.ent_embeds)  # initialize entity embeddings

        # the encoder: two-layer RGCN network
        self.rgcn1 = RGCN(in_dimension=dimension, out_dimension=dimension, num_relations=num_relations, num_bases=num_bases, aggr=aggr)
        self.rgcn2 = RGCN(in_dimension=dimension, out_dimension=dimension, num_relations=num_relations, num_bases=num_bases, aggr=aggr)

        # the decoder: DistMult
        self.distmult = Distmult(num_relations=num_relations, dimension=dimension)

    def encode(self, ent_ids: LongTensor, edge_index: LongTensor, edge_type: LongTensor) -> FloatTensor:
        x = torch.index_select(input=self.ent_embeds, index=ent_ids, dim=0)

        x = self.rgcn1.forward(x=x, edge_index=edge_index, edge_type=edge_type)
        x = functional.dropout(input=x, p=self.dropout, training=self.training)

        x = self.rgcn2.forward(x=x, edge_index=edge_index, edge_type=edge_type)
        x = functional.dropout(input=x, p=self.dropout, training=self.training)

        return x  # size: (num_entities_in_the_current_batch, dimension)

    def decode(self, x: FloatTensor, triples: LongTensor, lower_bound: FloatTensor, upper_bound: FloatTensor, mode: str) -> FloatTensor:
        head_ids = triples[:, 0]
        rel_ids = triples[:, 1]
        tail_ids = triples[:, 2]
        head_embeds = torch.index_select(input=x, index=head_ids, dim=0)  # head entity embeddings, size: (batch_size, dimension)
        tail_embeds = torch.index_select(input=x, index=tail_ids, dim=0)  # tail entity embeddings, size: (batch_size, dimension)
        scores = self.distmult(head_embeds=head_embeds, tail_embeds=tail_embeds, rel_ids=rel_ids, lower_bound=lower_bound, upper_bound=upper_bound, mode=mode)  # size: (batch_size)
        return scores


# the RGCN model
class RGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_relations: int, num_bases: int, aggr: str):
        super(RGCN, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr  # the aggregation scheme to use, "add" | "mean" | "max"

        self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation-specific coefficients
        torch.nn.init.xavier_normal_(self.coefficients)
        self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension, self.out_dimension))  # base matrices
        torch.nn.init.xavier_normal_(self.bases)

    def forward(self, x: FloatTensor, edge_index: LongTensor, edge_type: LongTensor) -> FloatTensor:
        # x: input entity embeddings, size: (num_entities, in_dimension);
        # edge_index: graph adjacency matrix in COO format, size: (2, num_edges);
        # edge_type: relation ids, the index is consistent with edge_index, size: (num_edges);
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type)  # propagate messages along edges and compute updated entity embeddings
        return torch.tanh(out)  # updated entity embeddings, size: (num_entities, out_dimension)

    def message(self, x_j: FloatTensor, edge_type: LongTensor = None) -> FloatTensor:
        # x_j: embeddings of source entities, size: (num_edges, in_dimension);
        relation_weights = torch.matmul(self.coefficients, self.bases.view(self.num_bases, -1)).view(self.num_relations, self.in_dimension, self.out_dimension)  # size: (num_relations, in_dimension, out_dimension)
        relation_weights = torch.index_select(input=relation_weights, dim=0, index=edge_type)  # size: (num_edges, in_dimension, out_dimension)
        messages = torch.bmm(x_j.unsqueeze(1), relation_weights).squeeze(1)  # unnormalized messages, size: (num_edges, out_dimension)
        return messages

    def aggregate(self, inputs: FloatTensor, index: LongTensor, edge_type: LongTensor = None) -> Tensor:
        # inputs: messages, size: (num_edges, out_dimension);
        # index: target entity ids, size: (num_edges);
        one_hot_relations_for_edges = functional.one_hot(edge_type, num_classes=self.num_relations).to(
            torch.float)  # the one-hot encoding of relation types, size: (num_edges, num_relations)
        relation_count_for_entities = torch_scatter.scatter(src=one_hot_relations_for_edges, index=index, dim=0, reduce="sum")  # relation count for all entities, size: (num_entities, num_relations); note: num_entities = max(entity ids in index) + 1;
        relation_count_for_edges = torch.index_select(input=relation_count_for_entities, index=index, dim=0)  # relation count for target entities, size: (num_target_entities, num_relations); note: num_target_entities == num_edges;
        relation_count = torch.gather(input=relation_count_for_edges, index=edge_type.view(-1, 1), dim=1)  # relation count for target entities selected according to edge_type, size: (num_target_entities, 1)
        normalization = 1. / torch.clip(input=relation_count, min=1.)  # normalization constants for target entities, size: (num_target_entities, 1)
        inputs = inputs * normalization  # normalized messages, size: (num_edges, out_dimension)
        return torch_scatter.scatter(src=inputs, index=index, dim=0, reduce=self.aggr)  # updated entity embeddings, size: (num_entities, out_dimension)


class Distmult(torch.nn.Module):
    def __init__(self, num_relations: int, dimension: int):
        super(Distmult, self).__init__()
        self.rel_embeds = torch.nn.Parameter(
            torch.FloatTensor(num_relations, dimension))  # the "diagonal" of relation matrices
        torch.nn.init.xavier_normal_(self.rel_embeds)

    def forward(self, head_embeds: FloatTensor, tail_embeds: FloatTensor, rel_ids: LongTensor, lower_bound: FloatTensor, upper_bound: FloatTensor, mode: str) -> FloatTensor:
        rel_embeds = torch.index_select(input=self.rel_embeds, index=rel_ids, dim=0)  # size: (batch_size, dimension)
        rel_matrices = torch.diag_embed(rel_embeds)  # size: (batch_size, dimension, dimension)
        scores = torch.bmm(
            torch.bmm(
                head_embeds.unsqueeze(1), rel_matrices
            ), tail_embeds.unsqueeze(2)
        ).view(-1)  # size: (batch_size)

        if mode == "train":
            pos_scores = scores[:upper_bound.size(0)]
            pos_scores = torch.minimum(pos_scores, upper_bound)

            neg_scores = scores[upper_bound.size(0):]
            neg_scores = torch.maximum(neg_scores, lower_bound)
            scores = torch.cat((pos_scores, neg_scores), dim=0)

        return torch.sigmoid(scores)
