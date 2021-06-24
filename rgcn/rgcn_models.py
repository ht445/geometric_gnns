import torch
from abc import ABC
import torch_scatter
import torch_geometric
from torch import Tensor
from typing import Optional
import torch.nn.functional as functional


# an implementation of the RGCN network*
# *. Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.
# note: self-loops should be added before calling the forward function
class RGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_relations: int, num_bases: int, aggr: str):
        super(RGCN, self).__init__()
        self.in_dimension = in_dimension  # the dimension of input entity embeddings
        self.out_dimension = out_dimension  # the dimension of output entity embeddings
        self.num_relations = num_relations  # the number of relations
        self.num_bases = num_bases  # the number of base matrices
        self.aggr = aggr  # the aggregation scheme to use

        self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension, self.out_dimension))  # base matrices
        torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
        self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation-specific coefficients
        torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

    # compute weights for given relation ids based on coefficients and bases
    # relation_ids: given relation ids, (num_given_relations)
    def compute_relation_weights(self, relation_ids: Tensor) -> Tensor:
        relation_coefficients = torch.index_select(input=self.coefficients, dim=0, index=relation_ids)  # relation coefficients corresponding to relation_ids, (num_given_relations, num_bases)
        relation_weights = torch.matmul(relation_coefficients, self.bases.view(self.num_bases, -1)).view(relation_ids.size()[0], self.in_dimension, self.out_dimension)
        # relation weights corresponding to relation_ids, (num_given_relations, in_dimension, out_dimension)
        return relation_weights

    # the rgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type)  # propagate messages along edges and compute updated entity embeddings
        return out  # updated entity embeddings, (num_entities, out_dimension)

    # compute messages along edges
    # by default, messages are from source entities (x_j) to target entities (x_i);
    # x_j: embeddings of source entities, (num_edges, in_dimension);
    # edge_type: as above, (num_edges);
    def message(self, x_j: Tensor, edge_type: Tensor = None) -> Tensor:
        assert edge_type is not None, "edge_type is not given"
        relation_weights = self.compute_relation_weights(edge_type)  # (num_relations, in_dimension, out_dimension)
        messages = torch.bmm(x_j.unsqueeze(1), relation_weights).squeeze(1)  # unnormalized messages, (num_edges, out_dimension)
        return messages

    # normalize and aggregate messages passed to target entities
    # inputs: messages, (num_edges, out_dimension);
    # index: target entity id list, (num_edges);
    # edge_type: as above, (num_edges);
    # ptr, dim_size: redundant parameters just to get rid of LSP violation warnings
    def aggregate(self, inputs: Tensor, index: Tensor, edge_type: Tensor = None, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        assert edge_type is not None, "edge_type is not given"

        one_hot_relations_for_edges = functional.one_hot(edge_type, num_classes=self.num_relations).to(torch.float)  # the one-hot encoding of relation types, (num_edges, num_relations)
        relation_count_for_entities = torch_scatter.scatter(src=one_hot_relations_for_edges, index=index, dim=0, reduce="sum")  # relation count for all entities, (num_entities, num_relations); note: num_entities = max(entity ids in index) + 1;
        relation_count_for_edges = torch.index_select(input=relation_count_for_entities, index=index, dim=0)  # relation count for target entities, (num_target_entities, num_relations); note: num_target_entities == num_edges;
        relation_count = torch.gather(input=relation_count_for_edges, index=edge_type.view(-1, 1), dim=1)  # relation count for target entities selected according to edge_type (num_target_entities, 1)
        normalization = 1. / torch.clip(input=relation_count, min=1.)  # normalization constants for target entities, (num_target_entities, 1)
        inputs = inputs * normalization  # normalized messages (num_edges, out_dimension)

        return torch_scatter.scatter(src=inputs, index=index, dim=0, reduce=self.aggr)  # updated target entity embeddings, (num_entities, out_dimension)

    # update target entity embeddings with an activation function
    def update(self, inputs: Tensor):
        return torch.nn.functional.leaky_relu(inputs)  # element-wise activation (num_entities, out_dimension)


# a link prediction network based on RGCN and DistMult *
# here we simply use the relation weights of RGCN as the relation matrices of DistMult (which violates some requirements of DistMult)
# *. Yang, Bishan, et al. "Embedding entities and relations for learning and inference in knowledge bases." arXiv preprint arXiv:1412.6575 (2014).
class RgcnLP(torch.nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int, num_entities: int, num_relations: int, num_bases: int, aggr: str):
        super(RgcnLP, self).__init__()
        assert in_dimension == out_dimension, "to use DistMult as the loss function, in_dimension should be equal to out_dimension"
        self.entity_embeds = torch.nn.Parameter(torch.FloatTensor(num_entities, in_dimension))
        torch.nn.init.kaiming_uniform_(self.entity_embeds, nonlinearity="leaky_relu")  # entity embeddings, (num_entities, in_dimension)

        self.rgcn = RGCN(in_dimension, out_dimension, num_relations, num_bases, aggr)  # the rgcn encoder

    # update entity embeddings via RGCN and compute scores via a DistMult-like function
    # triple_batch: positive and negative triples in the form of (head_entity_id, relation_id, tail_entity_id), (num_triples, 3);
    def forward(self, edge_index: Tensor, edge_type: Tensor, triple_batch: Tensor):
        # update entity embeddings via rgcn
        updated_entity_embeds = self.rgcn(x=self.entity_embeds, edge_index=edge_index, edge_type=edge_type)  # (num_entities, out_dimension)
        # compute loss via DistMult, note: in_dimension == out_dimension
        head_entity_embeds = torch.index_select(input=updated_entity_embeds, index=triple_batch[:, 0], dim=0)  # (num_triples, out_dimension)
        relation_matrices = self.rgcn.compute_relation_weights(relation_ids=triple_batch[:, 1])  # (num_triples, in_dimension, out_dimension)
        tail_entity_embeds = torch.index_select(input=updated_entity_embeds, index=triple_batch[:, 2], dim=0)  # (num_triples, out_dimension)
        scores = torch.bmm(
            torch.bmm(
                head_entity_embeds.unsqueeze(1), relation_matrices
            ), tail_entity_embeds.unsqueeze(2)
        ).view(-1)  # (num_triples)

        return torch.sigmoid(scores)


def test_rgcn(x, edge_index, edge_type):
    print("-- checking RGCN")
    print("input x:")
    print(x)
    rgcn = RGCN(in_dimension=6, out_dimension=5, num_relations=3, num_bases=2, aggr="add")
    x = rgcn(x, edge_index, edge_type)
    print("updated x:")
    print(x)
    print("")


def test_rgcnlp(edge_index, edge_type, triple_batch):
    print("-- checking RgcnLP")
    rgcn_lp = RgcnLP(in_dimension=6, out_dimension=6, num_entities=3, num_relations=3, num_bases=2, aggr="add")
    loss = rgcn_lp(edge_index, edge_type, triple_batch)
    print("computed loss")
    print(loss)


# test the above implemented models, you can add breakpoints to check how they run
if __name__ == "__main__":
    # create a toy graph with 3 entities, 2 relations, and 2 edges
    x = torch.FloatTensor(3, 6)
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    edge_type = torch.LongTensor([0, 1])
    # add self-loops to the toy graph
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
    self_loop_id = torch.LongTensor([2])  # the id of self-loop
    edge_type = torch.cat((edge_type, self_loop_id.repeat(edge_index.size()[1] - edge_type.size()[0])), dim=0)
    # test rgcn
    test_rgcn(x, edge_index, edge_type)

    # create some positive and negative triples
    triple_batch = torch.LongTensor([[0, 0, 1], [0, 0, 2], [1, 1, 2]])
    # test rgcn_lp
    test_rgcnlp(edge_index, edge_type, triple_batch)
