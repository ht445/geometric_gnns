import os
import psutil
import torch
from abc import ABC
import torch_scatter
import torch_geometric
from typing import Optional
import torch.nn.functional as functional
from torch import Tensor, LongTensor, FloatTensor
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torchviz import make_dot, make_dot_from_trace


class RgcnLP(torch.nn.Module):
    def __init__(self, edge_index: LongTensor, edge_attr: LongTensor, num_entities: int, num_relations: int,
                 dimension: int, num_bases: int, aggr: str, dropout: float,
                 num_subgraphs: int, subgraph_batch_size: int, device: torch.device):
        super(RgcnLP, self).__init__()
        self.dropout = dropout  # dropout rate
        self.num_subgraphs = num_subgraphs  # number of partitioned subgraphs
        self.subgraph_batch_size = subgraph_batch_size  # number of subgraphs in each cluster batch

        # define the training knowledge graph
        self.ent_embeds = torch.nn.Parameter(torch.FloatTensor(num_entities, dimension)).to(device)  # entity embeddings, size: (num_entities, dimension)
        torch.nn.init.kaiming_uniform_(self.ent_embeds, nonlinearity="leaky_relu")  # initialize entity embeddings
        self.y = torch.arange(start=0, end=num_entities, step=1).to(device)  # global entity ids, size: (num_entities)
        self.edge_index = edge_index.to(device)  # head- and tail-entity ids, size: (2, num_triples)
        self.edge_attr = edge_attr.to(device)  # relation ids, size: (num_triples, 1)
        self.train_kg = Data(x=self.ent_embeds, y=self.y, edge_index=self.edge_index, edge_attr=self.edge_attr)  # the training graph

        # partition the training graph into subgraphs and creat the cluster loader
        self.cluster_data = ClusterData(data=self.train_kg, num_parts=self.num_subgraphs)
        self.cluster_loader = ClusterLoader(cluster_data=self.cluster_data, batch_size=self.subgraph_batch_size, shuffle=True)

        # the encoder: two-layer RGCN network
        self.rgcn1 = RGCN(in_dimension=dimension, out_dimension=dimension, num_relations=num_relations, num_bases=num_bases, aggr=aggr)
        self.rgcn2 = RGCN(in_dimension=dimension, out_dimension=dimension, num_relations=num_relations, num_bases=num_bases, aggr=aggr)

        # the decoder: DistMult
        self.distmult = Distmult(num_relations=num_relations, dimension=dimension)

    def forward(self, triples: LongTensor):
        ent_embeds = self.encode()
        scores = self.decode(ent_embeds=ent_embeds, head_ids=triples[:, 0], rel_ids=triples[:, 1], tail_ids=triples[:, 2])
        return scores

    def encode(self) -> FloatTensor:
        # encoded entity embeddings and global entity ids
        out_x = None  # size should be: (num_entities, dimension)
        out_y = None  # size should be: (num_entities)

        for cluster in self.cluster_loader:
            print("number of triples: {}, memory: {}".format(cluster.y.size(0), psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))
            tmp_x = cluster.x  # size: (num_batch_entities, dimension)
            tmp_y = cluster.y  # size: (num_batch_entities)
            tmp_edge_index = cluster.edge_index  # size: (2, num_batch_triples)
            tmp_edge_type = cluster.edge_attr.view(-1)  # size: (num_batch_triples)

            tmp_x = self.rgcn1(x=tmp_x, edge_index=tmp_edge_index, edge_type=tmp_edge_type)
            tmp_x = functional.leaky_relu(tmp_x)
            tmp_x = functional.dropout(input=tmp_x, p=self.dropout, training=self.training)
            tmp_x = self.rgcn2(x=tmp_x, edge_index=tmp_edge_index, edge_type=tmp_edge_type)
            if out_x is None:
                out_x = tmp_x
                out_y = tmp_y
            else:
                out_x = torch.cat((out_x, tmp_x), dim=0)
                out_y = torch.cat((out_y, tmp_y), dim=0)
        out_x = torch_scatter.scatter(src=out_x, index=out_y, dim=0, reduce="mean")  # size: (num_entities, dimension)
        return out_x

    def decode(self, ent_embeds: FloatTensor, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        head_embeds = torch.index_select(input=ent_embeds, index=head_ids, dim=0)  # head entity embeddings, size: (batch_size, dimension)
        tail_embeds = torch.index_select(input=ent_embeds, index=tail_ids, dim=0)  # tail entity embeddings, size: (batch_size, dimension)
        scores = self.distmult(head_embeds=head_embeds, tail_embeds=tail_embeds, rel_ids=rel_ids)  # size: (batch_size)
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

        self.bases = torch.nn.Parameter(torch.FloatTensor(self.num_bases, self.in_dimension, self.out_dimension))  # base matrices
        torch.nn.init.kaiming_uniform_(self.bases, nonlinearity="leaky_relu")
        self.coefficients = torch.nn.Parameter(torch.FloatTensor(self.num_relations, self.num_bases))  # relation-specific coefficients
        torch.nn.init.kaiming_uniform_(self.coefficients, nonlinearity="leaky_relu")

    def compute_relation_weights(self, relation_ids: LongTensor) -> FloatTensor:
        relation_coefficients = torch.index_select(input=self.coefficients, dim=0, index=relation_ids)  # size: (num_relations, num_bases)
        relation_weights = torch.matmul(relation_coefficients, self.bases.view(self.num_bases, -1)).view(relation_ids.size()[0], self.in_dimension, self.out_dimension)  # size: (num_relations, in_dimension, out_dimension)
        return relation_weights

    def forward(self, x: FloatTensor, edge_index: LongTensor, edge_type: LongTensor) -> FloatTensor:
        # x: input entity embeddings, size: (num_entities, in_dimension);
        # edge_index: graph adjacency matrix in COO format, size: (2, num_edges);
        # edge_type: relation ids, the index is consistent with edge_index, size: (num_edges);
        out = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type)  # propagate messages along edges and compute updated entity embeddings
        return out  # updated entity embeddings, size: (num_entities, out_dimension)

    def message(self, x_j: FloatTensor, edge_type: LongTensor = None) -> FloatTensor:
        # x_j: embeddings of source entities, size: (num_edges, in_dimension);
        relation_weights = self.compute_relation_weights(edge_type)  # size: (num_relations, in_dimension, out_dimension)
        messages = torch.bmm(x_j.unsqueeze(1), relation_weights).squeeze(1)  # unnormalized messages, size: (num_edges, out_dimension)
        return messages

    def aggregate(self, inputs: FloatTensor, index: LongTensor, edge_type: LongTensor = None, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        # inputs: messages, size: (num_edges, out_dimension);
        # index: target entity ids, size: (num_edges);
        # ptr, dim_size: redundant parameters just to get rid of LSP violation warnings
        one_hot_relations_for_edges = functional.one_hot(edge_type, num_classes=self.num_relations).to(torch.float)  # the one-hot encoding of relation types, size: (num_edges, num_relations)
        relation_count_for_entities = torch_scatter.scatter(src=one_hot_relations_for_edges, index=index, dim=0, reduce="sum")  # relation count for all entities, size: (num_entities, num_relations); note: num_entities = max(entity ids in index) + 1;
        relation_count_for_edges = torch.index_select(input=relation_count_for_entities, index=index, dim=0)  # relation count for target entities, size: (num_target_entities, num_relations); note: num_target_entities == num_edges;
        relation_count = torch.gather(input=relation_count_for_edges, index=edge_type.view(-1, 1), dim=1)  # relation count for target entities selected according to edge_type, size: (num_target_entities, 1)
        normalization = 1. / torch.clip(input=relation_count, min=1.)  # normalization constants for target entities, size: (num_target_entities, 1)
        inputs = inputs * normalization  # normalized messages, size: (num_edges, out_dimension)
        return torch_scatter.scatter(src=inputs, index=index, dim=0, reduce=self.aggr)  # updated entity embeddings, size: (num_entities, out_dimension)


class Distmult(torch.nn.Module):
    def __init__(self, num_relations: int, dimension: int):
        super(Distmult, self).__init__()
        self.rel_embeds = torch.nn.Parameter(torch.FloatTensor(num_relations, dimension))  # the "diagonal" of relation matrices

    def forward(self, head_embeds: FloatTensor, tail_embeds: FloatTensor, rel_ids: LongTensor) -> FloatTensor:
        rel_embeds = torch.index_select(input=self.rel_embeds, index=rel_ids, dim=0)  # size: (batch_size, dimension)
        rel_matrices = torch.diag_embed(rel_embeds)  # size: (batch_size, dimension, dimension)
        scores = torch.bmm(
            torch.bmm(
                head_embeds.unsqueeze(1), rel_matrices
            ), tail_embeds.unsqueeze(2)
        ).view(-1)  # size: (batch_size)
        return torch.sigmoid(scores)


# check the above models by running on a toy graph
def test_rgcnlp():
    # create a toy graph
    edge_index = torch.LongTensor([[3, 2, 1, 0, 3, 2, 1, 0], [2, 1, 0, 3, 3, 2, 1, 0]])
    edge_attr = torch.LongTensor([[2], [2], [1], [1], [0], [0], [0], [0]])
    pos_triples = torch.LongTensor([[3, 2, 2], [2, 2, 1], [1, 1, 0], [0, 1, 3], [3, 0, 3], [2, 0, 2], [1, 0, 1], [0, 0, 0]])  # positive triples
    pos_targets = torch.ones(pos_triples.size()[0])
    neg_triples = torch.LongTensor([[3, 2, 1], [3, 2, 3], [0, 1, 0], [0, 1, 2], [0, 0, 3], [1, 0, 2], [1, 0, 0], [2, 0, 0]])  # negative triples
    neg_targets = torch.zeros(neg_triples.size()[0])
    # train the model
    rgcnlp = RgcnLP(edge_index=edge_index, edge_attr=edge_attr, num_entities=4, num_relations=3, dimension=5, num_bases=2, aggr="add", dropout=0.2, num_subgraphs=2, subgraph_batch_size=1, device=torch.device("cpu"))
    optimizer = torch.optim.Adam(params=rgcnlp.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    rgcnlp.train()
    scores = rgcnlp(triples=torch.cat((pos_triples, neg_triples), dim=0))
    print(scores)
    for epoch in range(50):
        optimizer.zero_grad()
        scores = rgcnlp(triples=torch.cat((pos_triples, neg_triples), dim=0))
        loss = criterion(input=scores, target=torch.cat((pos_targets, neg_targets), dim=0))
        loss.backward()
        optimizer.step()
        print("-- epoch: {}, loss: {}--".format(epoch, loss))
        print(scores)
    # test the model
    rgcnlp.eval()
    scores = rgcnlp(triples=pos_triples)
    print("- test scores -")
    print(scores)
    # render the execution graph of the model
    dot = make_dot(scores, params=dict(rgcnlp.named_parameters()), show_attrs=True, show_saved=True)
    dot.format = 'png'
    dot.render('rgcn_test_arch')


if __name__ == "__main__":
    test_rgcnlp()
