import torch
from abc import ABC
import torch_scatter
import torch_geometric
from torch import Tensor
from typing import Optional
import torch.nn.functional as functional


class RGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_relations: int, num_bases: int, aggr: str):
        super(RGCN, self).__init__(aggr=aggr)
        self.in_dimension = in_dimension  # the dimension of input entity embeddings
        self.out_dimension = out_dimension  # the dimension of output entity embeddings
        self.num_relations = num_relations  # the number of relations
        self.num_bases = num_bases  # the number of base matrices
        self.aggr = aggr  # the aggregation scheme to use
        self.bases = torch.nn.Parameter(
            torch.FloatTensor(self.num_bases, self.in_dimension, self.out_dimension)
        )  # base matrices
        torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
        self.coefficients = torch.nn.Parameter(
            torch.FloatTensor(self.num_relations, self.num_bases)
        )  # relation-specific coefficients
        torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

    # the rgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type)
        # propagate messages along edges and compute updated entity embeddings
        return out  # updated entity embeddings, (num_entities, out_dimension)

    # compute messages along edges
    # by default, messages are from source entities (x_j) to target entities (x_i);
    # x_j: embeddings of source entities, (num_edges, in_dimension);
    # edge_type: as above, (num_edges);
    def message(self, x_j: Tensor, edge_type: Tensor = None) -> Tensor:
        assert edge_type is not None, "edge_type is not given"

        relation_coefficients = torch.index_select(
            input=self.coefficients, dim=0, index=edge_type
        )  # relation coefficients corresponding to edge_type, (num_edges, num_bases)

        relation_weights = torch.matmul(
            relation_coefficients, self.bases.view(self.num_bases, -1)
        ).view(edge_type.size()[0], self.in_dimension, self.out_dimension)
        # relation weights corresponding to edge_type, (num_edges, in_dimension, out_dimension)

        messages = torch.bmm(
            x_j.unsqueeze(1), relation_weights
        ).squeeze(1)  # unnormalized messages, (num_edges, out_dimension)

        return messages

    # normalize and aggregate messages passed to target entities
    # inputs: messages, (num_edges, out_dimension);
    # index: target entity id list, (num_edges);
    # edge_type: as above, (num_edges);
    # ptr, dim_size: redundant parameters just to get rid of LSP violation warnings
    def aggregate(self, inputs: Tensor, index: Tensor, edge_type: Tensor = None, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        assert edge_type is not None, "edge_type is not given"

        one_hot_relations_for_edges = functional.one_hot(
            edge_type, num_classes=self.num_relations
        ).to(torch.float)  # the one-hot encoding of relation types, (num_edges, num_relations)

        relation_count_for_entities = torch_scatter.scatter(
            src=one_hot_relations_for_edges, index=index, dim=0, reduce="sum"
        )  # relation count for all entities, (num_entities, num_relations);
        # note: num_entities = max(entity ids in index) + 1;

        relation_count_for_edges = torch.index_select(
            input=relation_count_for_entities, index=index, dim=0
        )  # relation count for target entities, (num_target_entities, num_relations);
        # note: num_target_entities == num_edges;

        relation_count = torch.gather(
            input=relation_count_for_edges, index=edge_type.view(-1, 1), dim=1
        )  # relation count for target entities selected according to edge_type (num_target_entities, 1)

        normalization = 1. / torch.clip(input=relation_count, min=1.)
        # normalization constants for target entities, (num_target_entities, 1)

        inputs = inputs * normalization  # normalized messages (num_edges, out_dimension)

        return torch_scatter.scatter(src=inputs, index=index, dim=0, reduce=self.aggr)
        # updated target entity embeddings, (num_entities, out_dimension)

    # update target entity embeddings with an activation function
    def update(self, inputs: Tensor):
        return torch.nn.functional.leaky_relu(inputs)  # element-wise activation (num_entities, out_dimension)


def test_rgcn():
    # create a toy graph
    x = torch.ones(3, 6)
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    edge_type = torch.LongTensor([0, 1])
    # add self-loops
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
    self_loop_id = torch.LongTensor([2])
    edge_type = torch.cat((edge_type, self_loop_id.repeat(edge_index.size()[1] - edge_type.size()[0])), dim=0)

    print("original x:")
    print(x)
    rgcn = RGCN(in_dimension=6, out_dimension=4, num_relations=3, num_bases=2, aggr="add")
    x = rgcn(x, edge_index, edge_type)
    print("updated x:")
    print(x)


if __name__ == "__main__":
    print("-- just for testing --")
    test_rgcn()
