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

    # store training/validation/testing triples in tensors
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

    #  store training triples in dictionaries (would be used in negative sampling and filtered ranking)
    hr2t = {}  # {(head_entity_id, relation_id): [tail_entity_ids, ...]}
    tr2h = {}  # {(tail_entity_id, relation_id): [head_entity_ids, ...]}
    with open(data_path + "train2id.txt") as f:
        f.readline()
        line = f.readline()
        while line:
            head, tail, relation = line.rstrip("\n").split(" ")
            head, tail, relation = int(head), int(tail), int(relation)
            if (head, relation) not in hr2t:
                hr2t[(head, relation)] = []
            hr2t[(head, relation)].append(tail)
            if (tail, relation) not in tr2h:
                tr2h[(tail, relation)] = []
            tr2h[(tail, relation)].append(head)
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
    edge_type = torch.cat((edge_type, edge_type + count["relation"]), dim=0)
    count["relation"] = count["relation"] * 2

    # add self-loops
    self_loop_id = torch.LongTensor([count["relation"]])  # the id of self-loops
    count["relation"] += 1
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index=edge_index, num_nodes=count["entity"])
    edge_type = torch.cat((edge_type, self_loop_id.repeat(edge_index.size()[1] - edge_type.size()[0])), dim=0)
    # construct a pytorch geometric graph for the training graph
    graph = Data(edge_index=edge_index, edge_type=edge_type)

    # mark correct heads and tails in the tensors that would be used to filter out their scores
    correct_heads = {"valid": torch.LongTensor(count["valid"], count["entity"]), "test": torch.LongTensor(count["test"], count["entity"])}
    correct_tails = {"valid": torch.LongTensor(count["valid"], count["entity"]), "test": torch.LongTensor(count["test"], count["entity"])}
    for name in ["valid", "test"]:
        current_triples = triples[name]  # (num_valid/test_triples, 3)
        for i in range(count[name]):
            current_triple = current_triples[i, :]  # (3)
            current_head, current_relation, current_tail = int(current_triple[0]), int(current_triple[1]), int(current_triple[2])
            correct_heads[name][i, :] = torch.arange(count["entity"])
            if (current_tail, current_relation) in tr2h:
                for correct_head in tr2h[(current_tail, current_relation)]:
                    correct_heads[name][i, correct_head] = current_head
            correct_tails[name][i, :] = torch.arange(count["entity"])
            if (current_head, current_relation) in hr2t:
                for correct_tail in hr2t[(current_head, current_relation)]:
                    correct_tails[name][i, correct_tail] = current_tail

    return count, triples, graph, hr2t, tr2h, correct_heads, correct_tails


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
# hr2t: {(head_entity_id, relation_id): [tail_entity_ids, ...]}
# tr2h: {(tail_entity_id, relation_id): [head_entity_ids, ...]}
def negative_sampling(num_entities: int, num_triples: int, neg_num: int, train_triples: Tensor, hr2t: dict, tr2h: dict) -> Tensor:
    neg_triples = torch.LongTensor(num_triples, neg_num, 3)
    head_or_tail_sampler = torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=2), replacement=True, num_samples=num_triples * neg_num)  # if 0, corrupt head; if 1, corrupt tail;
    corrupt_entity_sampler = torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=num_entities), replacement=True, num_samples=num_triples * neg_num)  # sampled corrupt entities
    ht_samples = list(head_or_tail_sampler)  # int list, len(num_triples * neg_num)
    en_samples = list(corrupt_entity_sampler)  # int list, len(num_triples * neg_num)
    tmp_count = 0
    for t_id in range(num_triples):
        current_triple = train_triples[t_id, :]  # a positive triple, LongTensor(3)
        current_head, current_relation, current_tail = int(current_triple[0]), int(current_triple[1]), int(current_triple[2])
        for i in range(neg_num):
            sampled_position = ht_samples[tmp_count]
            sampled_entity = en_samples[tmp_count]
            if sampled_position == 0:  # corrupt the head
                while sampled_entity == current_head or sampled_entity in tr2h[(current_tail, current_relation)]:
                    sampled_entity += 1
                neg_triples[t_id, i, :] = torch.LongTensor([sampled_entity, current_relation, current_tail])
            elif sampled_position == 1:  # corrupt the tail
                while sampled_entity == current_tail or sampled_entity in hr2t[(current_head, current_relation)]:
                    sampled_entity += 1
                neg_triples[t_id, i, :] = torch.LongTensor([current_head, current_relation, sampled_entity])
            tmp_count += 1
    return neg_triples

