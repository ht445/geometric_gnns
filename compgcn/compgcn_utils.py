import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor


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

    #  store all positive triples in dictionaries
    hr2t = defaultdict(list) # {(head_entity_id, relation_id): [tail_entity_ids, ...]}
    tr2h = defaultdict(list)  # {(tail_entity_id, relation_id): [head_entity_ids, ...]}
    for name in ["train", "valid", "test"]:
        with open(data_path + name + "2id.txt") as f:
            f.readline()
            line = f.readline()
            while line:
                head, tail, relation = line.rstrip("\n").split(" ")
                head, tail, relation = int(head), int(tail), int(relation)

                hr2t[(head, relation)].append(tail)
                hr2t[(tail, relation + count["relation"])].append(head)

                tr2h[(tail, relation)].append(head)
                tr2h[(head, relation + count["relation"])].append(tail)
                line = f.readline()

    # mask correct heads and tails in tensors that would be used to compute filtered results
    correct_heads = {"valid": torch.LongTensor(count["valid"], count["entity"]),
                     "test": torch.LongTensor(count["test"], count["entity"])}
    correct_tails = {"valid": torch.LongTensor(count["valid"], count["entity"]),
                     "test": torch.LongTensor(count["test"], count["entity"])}
    for name in ["valid", "test"]:
        current_triples = triples[name]  # size: (num_valid/test_triples, 3)
        for i in range(count[name]):
            current_triple = current_triples[i, :]  # size: (3)
            current_head, current_relation, current_tail = int(current_triple[0]), int(current_triple[1]), int(current_triple[2])
            correct_heads[name][i, :] = torch.arange(count["entity"])
            if (current_tail, current_relation) in tr2h:
                for correct_head in tr2h[(current_tail, current_relation)]:
                    correct_heads[name][i, correct_head] = current_head
            correct_tails[name][i, :] = torch.arange(count["entity"])
            if (current_head, current_relation) in hr2t:
                for correct_tail in hr2t[(current_head, current_relation)]:
                    correct_tails[name][i, correct_tail] = current_tail

    return count, triples, correct_heads, correct_tails

# include inverse edges as training triples
def train_triple_pre_all(ent_ids: LongTensor, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor, neg_num: int, self_rel_id: int) -> [LongTensor]:
    num_pos_triples = head_ids.size(0)
    for edge_id in range(rel_ids.size(0)):
        if int(rel_ids[edge_id]) ==  self_rel_id:
            num_pos_triples -= 1
    pos_triples = torch.LongTensor(num_pos_triples, 3)  # positive training triples
    hrt2e = {}  # record positive triples
    triple_id = 0
    for edge_id in range(head_ids.size(0)):
        h_id = int(head_ids[edge_id])  # cluster head id
        r_id = int(rel_ids[edge_id])  # global relation id
        t_id = int(tail_ids[edge_id])  # cluster tail id
        if r_id != self_rel_id:
            pos_triples[triple_id] = torch.LongTensor([h_id, r_id, t_id])
            hrt2e[(h_id, r_id, t_id)] = None
            triple_id += 1
    assert triple_id == num_pos_triples, "triple_id: {}, num_pos_triples: {}, failed to collate all training triples; check train_triple_pre() in compgcn_utils.py".format(triple_id, num_pos_triples)

    pos_triples = pos_triples.repeat(neg_num, 1)
    num_pos_triples = num_pos_triples * neg_num

    neg_triples = torch.LongTensor(num_pos_triples, 3)  # negative triples
    h_or_ts = torch.LongTensor(num_pos_triples).random_(0, 2)  # if 0, corrupt the head; if 1, corrupt the tail;
    corr_ents = torch.LongTensor(num_pos_triples).random_(0, ent_ids.size(0))  # sampled corrupt entities
    neg_triple_count = 0
    for triple_id in range(num_pos_triples):
        h_id = int(pos_triples[triple_id][0])  # cluster entity id
        r_id = int(pos_triples[triple_id][1])  # global relation id
        t_id = int(pos_triples[triple_id][2])  # cluster entity id
        sampled_position = int(h_or_ts[triple_id])
        sampled_entity = int(corr_ents[triple_id])  # cluster entity id
        if sampled_position == 0:  # corrupt the head
            while (sampled_entity, r_id, t_id) in hrt2e:
                sampled_entity = (sampled_entity + 1) % ent_ids.size(0)
            neg_triples[triple_id] = torch.LongTensor([sampled_entity, r_id, t_id])
            neg_triple_count += 1
        elif sampled_position == 1:  # corrupt the tail
            while (h_id, r_id, sampled_entity) in hrt2e:
                sampled_entity = (sampled_entity + 1) % ent_ids.size(0)
            neg_triples[triple_id] = torch.LongTensor([h_id, r_id, sampled_entity])
            neg_triple_count += 1
    assert neg_triple_count == num_pos_triples, "failed to assemble negative triples; check train_triple_pre() in compgcn_utils.py"
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