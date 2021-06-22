import torch
import torch_geometric
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data


def read_data(data_path: str):
    # get the number of entities, relations, and training/validation/testing triples
    count = {}
    for name in ["entity", "relation", "train", "valid", "test"]:
        with open(data_path + name + "2id.txt") as f:
            num = f.readline()
            count[name] = int(num)

    # store training/validation/testing triples in dictionaries
    triples = {}
    for name in ["train", "valid", "test"]:
        triples[name] = torch.LongTensor(count[name], 3)  # (num_train/valid/test_triples, 3)
        tmp_count = 0
        with open(data_path + name + "2id.txt") as f:
            f.readline()
            line = f.readline()
            while line:
                ids = line.rstrip("\n").split(" ")
                triples[name][tmp_count, :] = torch.LongTensor([int(ids[0]), int(ids[2]), int(ids[1])])
                tmp_count += 1
                line = f.readline()

    # convert training triples into edge_index and edge_type
    edge_index = torch.LongTensor(2, count["train"])  # (2, num_train_triples)
    edge_type = torch.LongTensor(count["train"])  # (num_train_triples)
    for t_id in range(count["train"]):
        ids = triples["train"][t_id, :]
        edge_index[0, t_id] = int(ids[0])
        edge_type[t_id] = int(ids[1])
        edge_index[1, t_id] = int(ids[2])

    # add inverse relations
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_index = torch.cat((edge_index, torch.cat((targets.unsqueeze(0), sources.unsqueeze(0)), dim=0)), dim=1)
    edge_type = torch.cat((edge_type, edge_type), dim=0)

    # add self-loops
    self_loop_id = torch.LongTensor([count["relation"]])  # the id of self-loops
    count["relation"] += 1
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index=edge_index, num_nodes=count["entity"])
    edge_type = torch.cat((edge_type, self_loop_id.repeat(edge_index.size()[1] - edge_type.size()[0])), dim=0)
    # construct a pytorch geometric graph for the training graph
    graph = Data(edge_index=edge_index, edge_type=edge_type)

    return count, triples, graph


class IndexSet(Dataset):
    def __init__(self, num_indices):
        super(IndexSet, self).__init__()
        self.num_indices = num_indices
        self.index_list = torch.arange(self.num_indices, dtype=torch.long)

    def __len__(self):
        return self.num_indices

    def __getitem__(self, item):
        return self.index_list[item]


# num_entities: number of entities
# num_triples: number of training triples
# neg_num: number of negative triples for each training triple
# train_triples: LongTensor(num_train_triples, 3)
def negative_sampling(num_entities: int, num_triples: int, neg_num: int, train_triples: Tensor) -> Tensor:
    neg_triples = torch.LongTensor(num_triples, neg_num, 3)
    head_or_tail_sampler = torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=2), replacement=True, num_samples=neg_num)  # if 0, corrupt head; if 1, corrupt tail;
    corrupt_entity_sampler = torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=num_entities), replacement=True, num_samples=neg_num)  # sampled corrupt entities
    for t_id in range(num_triples):
        current_triple = train_triples[t_id, :]  # a positive triple, LongTensor(3)
        ht_samples = list(head_or_tail_sampler)  # int list, len(neg_num)
        en_samples = list(corrupt_entity_sampler)  # int list, len(neg_num)
        for i in range(neg_num):
            while en_samples[i] == int(current_triple[0]) or en_samples[i] == int(current_triple[2]):
                en_samples[i] += 1
            if ht_samples[i] == 0:
                neg_triples[t_id, i, :] = torch.LongTensor([en_samples[i], current_triple[1], current_triple[2]])
            elif ht_samples[i] == 1:
                neg_triples[t_id, i, :] = torch.LongTensor([current_triple[0], current_triple[1], en_samples[i]])
    return neg_triples

