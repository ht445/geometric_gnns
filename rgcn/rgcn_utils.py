import torch
import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset


def read_data(data_path: str) -> []:
    # get the number of entities, relations, and training/validation/testing triples
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

    return count, triples, hr2t, tr2h


def train_triple_pre(ent_ids: LongTensor, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor, hr2t: dict, tr2h: dict, neg_num: int) -> [LongTensor]:
    # prepare positive triples
    num_ori_triples = int((head_ids.size(0) - ent_ids.size(0)) / 2)
    pos_triples = torch.LongTensor(num_ori_triples, 3)
    triple_id = 0
    for edge_id in range(head_ids.size(0)):
        h_id = int(ent_ids[head_ids[edge_id]])
        r_id = int(rel_ids[edge_id])
        t_id = int(ent_ids[tail_ids[edge_id]])
        if (h_id, r_id) in hr2t:
            if t_id in hr2t[(h_id, r_id)]:
                pos_triples[triple_id] = torch.LongTensor([head_ids[edge_id], r_id, tail_ids[edge_id]])
                triple_id += 1
    # prepare negative triples
    neg_triples = torch.LongTensor(num_ori_triples * neg_num, 3)
    h_or_ts = list(torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=2), replacement=True, num_samples=num_ori_triples * neg_num))  # if 0, corrupt the head; if 1, corrupt the tail;
    corr_ents = list(torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=ent_ids.size(0)), replacement=True, num_samples=num_ori_triples * neg_num))  # sampled corrupt entities
    for triple_id in range(num_ori_triples):
        h_id = int(pos_triples[triple_id][0])
        r_id = int(pos_triples[triple_id][1])
        t_id = int(pos_triples[triple_id][2])
        for i in range(neg_num):
            sampled_position = int(h_or_ts[triple_id * neg_num + i])
            sampled_entity = int(corr_ents[triple_id * neg_num + i])
            if sampled_position == 0:  # corrupt the head
                while sampled_entity == h_id or int(ent_ids[sampled_entity]) in tr2h[(int(ent_ids[t_id]), r_id)]:
                    sampled_entity = (sampled_entity + 1) % ent_ids.size(0)
                neg_triples[triple_id * neg_num + i, :] = torch.LongTensor([sampled_entity, r_id, t_id])
            elif sampled_position == 1:  # corrupt the tail
                while sampled_entity == t_id or int(ent_ids[sampled_entity]) in hr2t[(int(ent_ids[h_id]), r_id)]:
                    sampled_entity = (sampled_entity + 1) % ent_ids.size(0)
                neg_triples[triple_id * neg_num + i, :] = torch.LongTensor([h_id, r_id, sampled_entity])
    return pos_triples, neg_triples


class IndexSet(Dataset):
    def __init__(self, num_indices):
        super(IndexSet, self).__init__()
        self.num_indices = num_indices
        self.index_list = torch.arange(self.num_indices, dtype=torch.long)

    def __len__(self):
        return self.num_indices

    def __getitem__(self, item):
        return self.index_list[item]