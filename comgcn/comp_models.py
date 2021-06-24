import torch
from abc import ABC
import torch_geometric
from torch import Tensor


# an implementation of the CompGCN network (with subtraction as the composition operator)*
# *. Vashishth, Shikhar, et al. "Composition-based Multi-Relational Graph Convolutional Networks." International Conference on Learning Representations. 2020.
class CompGCN(torch_geometric.nn.MessagePassing, ABC):
    def __init__(self, in_dimension: int, out_dimension: int, num_bases: int, num_relations: int, aggr: str, first_layer: bool):
        super(CompGCN, self).__init__(aggr=aggr)

        self.in_dimension = in_dimension  # the dimension of input entity/relation embeddings
        self.out_dimension = out_dimension  # the dimension of output entity/relation embeddings
        self.num_bases = num_bases  # the number of relation embedding bases
        self.num_relations = num_relations  # the number of relations
        self.first_layer = first_layer  # if true, relation embeddings are computed with bases; otherwise, they are directly provided by previous layers

        # base matrices and coefficients for relation embedding computation
        if first_layer:
            self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension))  # base vectors for relations
            torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
            self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation coefficients
            torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

        # the weights for original, inverse, and self-loop relations: weights[0, :, :], weights[1, :, :], weights[2, :, :]
        self.weights = torch.nn.Parameter(torch.FloatTensor(3, self.in_dimension, self.out_dimension))
        torch.nn.init.kaiming_uniform_(self.weights, nonlinearity="leaky_relu")

        # the weight for updating relations
        self.relation_weight = torch.nn.Parameter(torch.FloatTensor(self.in_dimension, self.out_dimension))  # the weight for relation transformation
        torch.nn.init.kaiming_uniform_(self.relation_weight)

    # the compgcn network forward computation
    # x: input entity embeddings, (num_entities, in_dimension);
    # edge_index: graph adjacency matrix in COO format, (2, num_edges);
    # edge_type: relation id list, and the order corresponds to edge_index, (num_edges);
    # masks: relation type list: original, inverse, or self-loop, (num_edges)
    # r: input relation embeddings (None when this is the first layer), (num_relations, in_dimension);
    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor, masks: Tensor = None, r: Tensor = None) -> [Tensor, Tensor]:
        if r is None:
            relation_coefficients = torch.index_select(input=self.coefficients, dim=0, index=torch.arange(self.num_relations))  # relation coefficients, (num_relations, num_bases)
            r = torch.matmul(relation_coefficients, self.bases)  # relation embeddings, (num_relations, in_dimension)
        rel_embeds = torch.index_select(input=r, dim=0, index=edge_type)  # (num_edges, in_dimension)
        updated_x = self.propagate(x=x, edge_index=edge_index, rel_embeds=rel_embeds, masks=masks)  # propagate messages along edges and compute updated entity embeddings
        updated_r = torch.matmul(r, self.relation_weight)  # (num_relations, out_dimension)
        return updated_x, updated_r  # updated entity embeddings, (num_entities, out_dimension)

    # compute messages along edges; by default, messages are from source entities (x_j) to target entities (x_i);
    # x_j: embeddings of source entities, (num_edges, in_dimension);
    # rel_embeds: as above, (num_edges, in_dimension);
    # masks: as above, (num_edges)
    def message(self, x_j: Tensor, rel_embeds: Tensor = None, masks: Tensor = None) -> Tensor:
        edge_weights = torch.index_select(input=self.weights, dim=0, index=masks)  # (num_edges, in_dimension, out_dimension)
        messages = x_j + rel_embeds  # (num_edges, in_dimension)
        messages = torch.bmm(messages.unsqueeze(1), edge_weights).squeeze(1)  # (num_edges, out_dimension)
        return messages

    # inputs: the results of aggregation with scheme aggr, (num_entities, out_dimension)
    # update target entity embeddings with an activation function
    def update(self, inputs: Tensor):
        return torch.nn.functional.leaky_relu(inputs)  # element-wise activation (num_entities, out_dimension)


def test_rgcn(x, edge_index, edge_type, masks):
    print("-- checking compgcn --")
    print("input x:")
    print(x)
    rgcn = CompGCN(in_dimension=6, out_dimension=5, num_relations=3, num_bases=2, aggr="add", first_layer=True)
    x, r = rgcn(x=x, edge_index=edge_index, edge_type=edge_type, masks=masks)
    print("updated x:")
    print(x)
    print("updated r:")
    print(r)
    print("")


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
    edge_type = torch.cat((edge_type, edge_type), dim=0)
    masks = torch.cat((masks, torch.LongTensor([1]).repeat(num_original_edges)), dim=0)
    # add self-loops
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=3)
    num_self_loops = edge_index.size()[1] - edge_type.size()[0]
    self_loop_id = torch.LongTensor([2])  # the id of self-loop
    edge_type = torch.cat((edge_type, self_loop_id.repeat(num_self_loops)), dim=0)
    masks = torch.cat((masks, torch.LongTensor([2]).repeat(num_self_loops)), dim=0)
    # test rgcn
    test_compgcn(x, edge_index, edge_type, masks)

    '''
    # create some positive and negative triples
    triple_batch = torch.LongTensor([[0, 0, 1], [0, 0, 2], [1, 1, 2]])
    # test rgcn_lp
    test_rgcnlp(edge_index, edge_type, triple_batch)
    '''





