from abc import ABC

import torch
import torch_geometric
from torch import Tensor


# an implementation of the CompGCN network (with subtraction as the composition operator)*
# *. Vashishth, Shikhar, et al. "Composition-based Multi-Relational Graph Convolutional Networks." International Conference on Learning Representations. 2020.
# note: self-loops should be added before calling the forward function
class CompGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_bases: int, num_relations: int, aggr: str, first_layer: bool):
        super(CompGCN, self).__init__()

        self.in_dimension = in_dimension  # the dimension of input entity/relation embeddings
        self.out_dimension = out_dimension  # the dimension of output entity/relation embeddings
        self.num_bases = num_bases  # the number of relation embedding bases
        self.num_relations = num_relations  # the number of relations
        self.aggr = aggr  # the aggregation scheme to use
        self.first_layer = first_layer  # if true, relation embeddings are computed with bases; otherwise, they are directly provided by previous layers

        self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension))  # base vectors for relations
        torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
        self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation coefficients
        torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

        self.original_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, self.out_dimension))  # the weight for original relations
        torch.nn.init.kaiming_uniform_(self.original_weight)
        self.inverse_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, self.out_dimension))  # the weight for inverse relations
        torch.nn.init.kaiming_uniform_(self.inverse_weight)
        self.loop_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, self.out_dimension))  # the weight for self-loops
        torch.nn.init.kaiming_uniform_(self.loop_weight)

        self.relation_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, self.out_dimension))  # the weight for relation transformation
        torch.nn.init.kaiming_uniform_(self.relation_weight)

    # compute relation embeddings for given relation ids based on coefficients and bases
    # relation_ids: given relation ids, (num_given_relations)
    def compute_relation_embeds(self, relation_ids: Tensor) -> Tensor:
        relation_coefficients = torch.index_select(input=self.coefficients, dim=0, index=relation_ids)   # relation coefficients corresponding to relation_ids, (num_given_relations, num_bases)
        relation_weights = torch.matmul(relation_coefficients, self.bases)  # relation embeddings corresponding to relation_ids, (num_given_relations, in_dimension)
        return relation_weights

    # the compgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    # r: input relation embeddings (None when this is the first layer), (num_relations, in_dimension);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor, r: Tensor = None) -> Tensor:
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, r=r)  # propagate messages along edges and compute updated entity embeddings
        return out  # updated entity embeddings, (num_entities, out_dimension)
