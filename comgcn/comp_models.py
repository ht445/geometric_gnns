import torch
from abc import ABC
import torch_geometric
import torch_scatter
from torch import Tensor
from typing import Optional
import torch.nn.functional as functional


# an implementation of the CompGCN network (with subtraction as the composition operator)*
# *. Vashishth, Shikhar, et al. "Composition-based Multi-Relational Graph Convolutional Networks." International Conference on Learning Representations. 2020.
class CompGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_bases: int, num_relations: int, aggr: str, first_layer: bool):
        super(CompGCN, self).__init__(aggr=aggr)
        print("- CompGCN instantiated")
        # base vectors and coefficients for relation embedding computation
        if first_layer:
            self.bases = torch.nn.Parameter(torch.FloatTensor(num_bases, in_dimension))  # base vectors for relations
            torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
            self.coefficients = torch.nn.Parameter(torch.FloatTensor(num_relations, num_bases))  # relation coefficients
            torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

        # the weights for original, inverse, and self-loop relations: weights[0, :, :], weights[1, :, :], weights[2, :, :]
        self.weights = torch.nn.Parameter(torch.FloatTensor(3, in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.weights, nonlinearity="leaky_relu")

        # the weight for updating relations
        self.relation_weight = torch.nn.Parameter(torch.FloatTensor(in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.relation_weight, nonlinearity="leaky_relu")

    # the compgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    # masks: relation type list: 0:original, 1:inverse, or 2:self-loop, (num_edges)
    # r: input relation embeddings (None when this is the first layer), (num_relations, in_dimension);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor, masks: Tensor = None, r: Tensor = None) -> [Tensor, Tensor]:
        if r is None:
            r = torch.matmul(self.coefficients, self.bases)  # relation embeddings, (num_relations, in_dimension)
        edge_rel_embeds = torch.index_select(input=r, dim=0, index=edge_type)  # (num_edges, in_dimension)
        updated_x = self.propagate(x=x, edge_index=edge_index, edge_rel_embeds=edge_rel_embeds, masks=masks)  # propagate messages along edges and compute updated entity embeddings
        updated_r = torch.matmul(r, self.relation_weight)  # (num_relations, out_dimension)
        return updated_x, updated_r  # updated entity embeddings, [(num_entities, out_dimension), (num_relations, out_dimension)]

    # compute messages along edges; by default, messages are from source entities (x_j) to target entities (x_i);
    # x_j: embeddings of source entities, (num_edges, in_dimension);
    # edge_rel_embeds: relation embeddings corresponding to edge_type, (num_edges, in_dimension);
    # masks: as above, (num_edges)
    def message(self, x_j: Tensor, edge_rel_embeds: Tensor = None, masks: Tensor = None) -> Tensor:
        edge_weights = torch.index_select(input=self.weights, dim=0, index=masks)  # (num_edges, in_dimension, out_dimension)
        messages = x_j + edge_rel_embeds  # (num_edges, in_dimension)
        messages = torch.bmm(messages.unsqueeze(1), edge_weights).squeeze(1)  # (num_edges, out_dimension)
        return messages

    # inputs: the results of aggregation, (num_entities, out_dimension)
    # update entity embeddings with an activation function
    def update(self, inputs: Tensor):
        return torch.nn.functional.leaky_relu(inputs)  # element-wise activation (num_entities, out_dimension)


class CompRgcn(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_bases: int, num_relations: int, aggr: str, first_layer: bool):
        super(CompRgcn, self).__init__()
        print("- CompRGCN instantiated")
        self.num_relations = num_relations  # the number of relations
        self.in_dimension = in_dimension  # the dimension of input entity embeddings
        self.out_dimension = out_dimension  # the dimension of output entity embeddings
        self.num_bases = num_bases  # the number of bases for relation embeddings
        self.aggr = aggr  # the aggregation scheme
        # base matrices and coefficients for relation embedding computation
        if first_layer:
            self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension, self.in_dimension))  # base matrices for relations
            torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
            self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation coefficients
            torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

        # the weights for original, inverse, and self-loop relations: weights[0, :, :], weights[1, :, :], weights[2, :, :]
        self.weights = torch.nn.Parameter(torch.FloatTensor(3, self.in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.weights, nonlinearity="leaky_relu")

        # the weight for updating relations
        self.relation_r_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, out_dimension))
        torch.nn.init.kaiming_uniform_(self.relation_r_weight, nonlinearity="leaky_relu")
        self.relation_l_weight = torch.nn.Parameter(torch.FloatTensor(self.out_dimension, in_dimension))
        torch.nn.init.kaiming_uniform_(self.relation_l_weight, nonlinearity="leaky_relu")

    # the compgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    # masks: relation type list: 0:original, 1:inverse, or 2:self-loop, (num_edges)
    # r: input relation embeddings (None when this is the first layer), (num_relations, in_dimension, in_dimension);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor, masks: Tensor = None, r: Tensor = None) -> [Tensor, Tensor]:
        if r is None:
            r = torch.matmul(self.coefficients, self.bases.view(self.num_bases, -1)).view(self.num_relations, self.in_dimension, self.in_dimension)  # relation embeddings, (num_relations, in _dimension, in_dimension)
        edge_rel_embeds = torch.index_select(input=r, dim=0, index=edge_type)  # (num_edges, in_dimension, in_dimension)
        updated_x = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, edge_rel_embeds=edge_rel_embeds, masks=masks)  # propagate messages along edges and compute updated entity embeddings
        updated_r = torch.bmm(r, self.relation_r_weight.unsqueeze(0).repeat(self.num_relations, 1, 1))  # (num_relations, in_dimension, out_dimension)
        updated_r = torch.bmm(self.relation_l_weight.unsqueeze(0).repeat(self.num_relations, 1, 1), updated_r)  # (num_relations, out_dimension, out_dimension)
        return updated_x, updated_r  # updated entity embeddings, [(num_entities, out_dimension), (num_relations, out_dimension)]

    # compute messages along edges; by default, messages are from source entities (x_j) to target entities (x_i);
    # x_j: embeddings of source entities, (num_edges, in_dimension);
    # edge_rel_embeds: relation embeddings corresponding to edge_type, (num_edges, in_dimension, in_dimension);
    # masks: as above, (num_edges)
    def message(self, x_j: Tensor, edge_rel_embeds: Tensor = None, masks: Tensor = None) -> Tensor:
        edge_weights = torch.index_select(input=self.weights, dim=0, index=masks)  # (num_edges, in_dimension, out_dimension)
        messages = torch.bmm(x_j.unsqueeze(1), edge_rel_embeds).squeeze(1)  # (num_edges, in_dimension)
        messages = torch.bmm(messages.unsqueeze(1), edge_weights).squeeze(1)  # (num_edges, out_dimension)
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

    # inputs: the results of aggregation, (num_entities, out_dimension)
    # update entity embeddings with an activation function
    def update(self, inputs: Tensor):
        return torch.nn.functional.leaky_relu(inputs)  # element-wise activation (num_entities, out_dimension)


# a link prediction network based on CompGCN and TransE *
# Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." Neural Information Processing Systems (NIPS). 2013.
class CompgcnLP(torch.nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int, num_entities: int, num_relations: int, num_bases: int, aggr: str, norm: int, op: str):
        super(CompgcnLP, self).__init__()
        self.norm = norm
        self.op = op  # the aggregation function to use, in ["TransE", "RGCN"]
        self.entity_embeds = torch.nn.Parameter(torch.FloatTensor(num_entities, in_dimension))
        torch.nn.init.kaiming_uniform_(self.entity_embeds, nonlinearity="leaky_relu")  # entity embeddings, (num_entities, in_dimension)
        assert self.op in ["TransE", "RGCN"]
        # aggregation operation: TransE or RGCN
        if self.op == "TransE":
            self.compgcn = CompGCN(in_dimension=in_dimension, out_dimension=out_dimension, num_bases=num_bases, num_relations=num_relations, aggr=aggr, first_layer=True)  # the compGCN encoder
        else:
            self.compgcn = CompRgcn(in_dimension=in_dimension, out_dimension=out_dimension, num_bases=num_bases, num_relations=num_relations, aggr=aggr, first_layer=True)  # the compGCN encoder

    # update entity embeddings via compGCN and compute scores via the TransE subtraction
    # triple_batch: positive and negative triples in the form of (head_entity_id, relation_id, tail_entity_id), (num_triples, 3);
    def forward(self, edge_index: Tensor, edge_type: Tensor, masks: Tensor, triple_batch: Tensor):
        # update entity and relation embeddings via compGCN
        updated_entity_embeds, updated_relation_embeds = self.compgcn(x=self.entity_embeds, edge_index=edge_index, edge_type=edge_type, masks=masks, r=None)  # [(num_entities, out_dimension), (num_relations, out_dimension, out_dimension)]
        head_entity_embeds = torch.index_select(input=updated_entity_embeds, index=triple_batch[:, 0], dim=0)  # (num_triples, out_dimension)
        tail_entity_embeds = torch.index_select(input=updated_entity_embeds, index=triple_batch[:, 2], dim=0)  # (num_triples, out_dimension)
        # compute loss via TransE
        if self.op == "TransE":
            relation_embeds = torch.index_select(input=updated_relation_embeds, index=triple_batch[:, 1], dim=0)  # (num_triples, out_dimension)
            scores = torch.norm(head_entity_embeds + relation_embeds - tail_entity_embeds, dim=1, p=self.norm)  # (num_triples)
        else:
            relation_matrices = torch.index_select(input=updated_relation_embeds, index=triple_batch[:, 1], dim=0)  # (num_triples, out_dimension, out_dimension)
            scores = torch.bmm(
                torch.bmm(head_entity_embeds.unsqueeze(1), relation_matrices), tail_entity_embeds.unsqueeze(2)
            ).view(-1)  # (num_triples)

        return torch.sigmoid(scores)


def test_compgcn(x, edge_index, edge_type, masks):
    print("-- checking CompGCN")
    print("input x:")
    print(x)
    compgcn = CompGCN(in_dimension=6, out_dimension=6, num_relations=5, num_bases=2, aggr="add", first_layer=True)
    x, r = compgcn(x=x, edge_index=edge_index, edge_type=edge_type, masks=masks)
    print("updated x:")
    print(x)
    print("updated r:")
    print(r)
    print("")


def test_compgcnlp(edge_index, edge_type, masks, triple_batch):
    print("-- checking CompgcnLP")
    compgcn_lp = CompgcnLP(in_dimension=6, out_dimension=5, num_entities=3, num_relations=5, num_bases=2, aggr="add", norm=2, op="RGCN")
    loss = compgcn_lp(edge_index=edge_index, edge_type=edge_type, masks=masks, triple_batch=triple_batch)
    print("computed loss")
    print(loss)


# test the above implemented models, you can add breakpoints to check how they run
if __name__ == "__main__":
    # create a toy graph with 3 entities, 2 relations, and 2 edges
    x = torch.FloatTensor(3, 6)
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    edge_type = torch.LongTensor([0, 1])
    num_original_edges = edge_type.size()[0]
    masks = torch.LongTensor([0]).repeat(num_original_edges)
    # add inverse relations
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_index = torch.cat((edge_index, torch.cat((targets.unsqueeze(0), sources.unsqueeze(0)), dim=0)), dim=1)
    edge_type = torch.cat((edge_type, edge_type + 2), dim=0)
    masks = torch.cat((masks, torch.LongTensor([1]).repeat(num_original_edges)), dim=0)
    # add self-loops
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=3)
    num_self_loops = edge_index.size()[1] - edge_type.size()[0]
    self_loop_id = torch.LongTensor([4])  # the id of self-loop
    edge_type = torch.cat((edge_type, self_loop_id.repeat(num_self_loops)), dim=0)
    masks = torch.cat((masks, torch.LongTensor([2]).repeat(num_self_loops)), dim=0)
    # test compgcn
    test_compgcn(x, edge_index, edge_type, masks)
    # create some positive and negative triples
    triple_batch = torch.LongTensor([[0, 0, 1], [0, 0, 2], [1, 1, 2]])
    # test compgcn_lp
    test_compgcnlp(edge_index, edge_type, masks, triple_batch)





