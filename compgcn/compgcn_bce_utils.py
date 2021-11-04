import torch
import torch_geometric
from collections import defaultdict
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor


def read_data(data_path: str) -> []:
    # get the number of entities, relations, and original training/validation/testing triples
    count = {}
    for name in ["entity", "relation", "train", "valid", "test"]:
        with open(data_path + name + "2id.txt") as f:
            num = f.readline()
            count[name] = int(num)

    # store training/validation/testing triples in tensors
    triples = {}
    for name in ["train", "valid", "test"]:
        triples[name] = torch.LongTensor(count[name], 3)  # size: (num_train/valid/test_triples, 3)
        tmp_count = 0
        with open(data_path + name + "2id.txt") as f:
            f.readline()
            line = f.readline()
            while line:
                ids = line.rstrip("\n").split(" ")
                triples[name][tmp_count, :] = torch.LongTensor([int(ids[0]), int(ids[2]), int(ids[1])])
                tmp_count += 1
                line = f.readline()

    # create the training graph
    edge_index = torch.LongTensor(2, count["train"])  # head and tail entity ids (changes after partitioning)
    edge_attr = torch.LongTensor(count["train"], 1)  # relation ids (remains after partitioning)
    y = torch.zeros(count["train"], 1, dtype=torch.long)  # relation types (remains after partitioning), 0: original, 1: inverse, 2: self-edge
    for triple_id in range(count["train"]):
        ids = triples["train"][triple_id, :]
        edge_index[0, triple_id] = ids[0]
        edge_attr[triple_id, 0] = ids[1]
        edge_index[1, triple_id] = ids[2]

    # add inverse edges
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_index = torch.cat((edge_index, torch.cat((targets.unsqueeze(0), sources.unsqueeze(0)), dim=0)), dim=1)  # size: (2, num_train_triples * 2)
    edge_attr = torch.cat((edge_attr, edge_attr + count["relation"]), dim=0)  # size: (num_train_triples * 2, 1)
    y = torch.cat((y, torch.ones(count["train"], 1, dtype=torch.long)), dim=0)  # size: (num_train_triples * 2, 1)
    count["relation"] = count["relation"] * 2  # double the number of relations

    # add self-loops
    self_loop_id = torch.LongTensor([count["relation"]])  # id of the self-loop relation
    edge_index = torch.cat((edge_index, torch.cat((torch.arange(count["entity"]).unsqueeze(0), torch.arange(count["entity"]).unsqueeze(0)), dim=0)), dim=1)  # size: (2, num_train_triples * 2 + num_entities)
    edge_attr = torch.cat((edge_attr, self_loop_id.repeat(edge_index.size(1) - edge_attr.size(0), 1)), dim=0)  # size: (num_train_triples * 2 + num_entities, 1)
    y = torch.cat((y, torch.ones(edge_attr.size(0) - y.size(0), 1, dtype=torch.long) + 1), dim=0)  # size: (num_train_triples * 2 + num_entities, 1)
    count["relation"] += 1

    # construct a geometric data as the training graph
    x = torch.arange(count["entity"]).unsqueeze(1)  # use x to store original entity ids since entity ids in edge_index will change after partitioning, size: (num_entities, 1)
    graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    #  store positive training tails in a dictionary
    pos_tails = defaultdict(dict)  # {(head_entity_id, relation_id): {positive_tail_entity_ids: None}}, including inverse edges
    for edge_id in range(edge_index.size(1)):
        head = int(edge_index[0, edge_id])
        tail = int(edge_index[1, edge_id])
        rel = int(edge_attr[edge_id, 0])
        if int(y[edge_id, 0]) != 2:
            pos_tails[(head, rel)][tail] = None

    return count, triples, graph, pos_tails

def pre_train_triples(head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor, num_cluster_entities: int, self_rel_id: int) -> [FloatTensor]:
    num_train_triples = rel_ids.size(0) - torch.nonzero(rel_ids == self_rel_id).size(0)

    heads = torch.LongTensor(num_train_triples)  # head entities of training triples
    relations = torch.LongTensor(num_train_triples)  # relations of training triples
    targets = torch.ones(num_train_triples, num_cluster_entities, dtype=torch.float)  # training targets regarding all entities in the current cluster

    triple_count = 0
    for edge_id in range(head_ids.size(0)):
        r_id = int(rel_ids[edge_id])
        if r_id != self_rel_id:
            h_id = int(head_ids[edge_id])
            t_id = int(tail_ids[edge_id])
            heads[triple_count] = h_id
            relations[triple_count] = r_id
            targets[triple_count, t_id] = 0.
            triple_count += 1
    assert triple_count == num_train_triples, "error in pre_train_triples()"
    return heads, relations, targets

class IndexSet(Dataset):
    def __init__(self, num_indices):
        super(IndexSet, self).__init__()
        self.num_indices = num_indices
        self.index_list = torch.arange(self.num_indices, dtype=torch.long)

    def __len__(self):
        return self.num_indices

    def __getitem__(self, item):
        return self.index_list[item]