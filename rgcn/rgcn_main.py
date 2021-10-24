import torch
import pickle
import wandb
import torch_scatter
import torch_geometric
import numpy as np
from itertools import product
from datetime import datetime
from rgcn_models import RgcnLP
from torch.utils.data import DataLoader
from rgcn_utils import read_data, IndexSet, train_triple_pre_all


class RgcnMain:
    def __init__(self, lr: float, batch_size: int, margin: float):
        self.data_path = "../data/FB15K237/"
        self.model_path = "../pretrained/FB15K237/rgcn_lp.pt"

        self.from_pre = False  # True: continue training
        self.num_epochs = 50  # number of training epochs
        self.valid_freq = 1  # do validation every x training epochs
        self.lr = lr  # learning rate
        self.dropout = 0.2  # dropout rate
        self.margin = margin  # margin

        self.aggr = "add"  # aggregation scheme to use in RGCN, "add" | "mean" | "max"
        self.embed_dim = 100  # entity embedding dimension

        self.neg_num = 8  # number of negative triples for each positive triple
        self.num_bases = 50  # number of bases for relation matrices in RGCN

        self.num_subgraphs = 200  # partition the training graph into x subgraphs; please set it according to your GPU memory (if available)
        self.cluster_size = 24  # number of subgraphs in each batch

        self.batch_size = batch_size  # batch size
        self.vt_batch_size = 12  # validation/test batch size

        self.eval_sampling = False  # True: sample candidate entities in validation and test
        self.eval_sample_size = 10000  # sample x candidate entities in validation and test

        self.highest_mrr = 0.  # highest mrr in validation

        if torch.cuda.is_available():
            self.device = torch.device("cuda:4")
            self.eval_device = torch.device("cuda:5")
        else:
            self.device = torch.device("cpu")
            self.eval_device = torch.device("cpu")

        self.count = None  # {"entity": num_entities, "relation": num_relations, "train": num_train_triples, "valid": num_valid_triples, "test": num_test_triples};
        self.triples = None  # {"train": LongTensor(num_train_triples, 3), "valid": LongTensor(num_valid_triples, 3), "test": LongTensor(num_test_triples, 3)};
        self.hr2t = None  # {(head_entity_id, relation_id): [tail_entity_ids, ...]}
        self.tr2h = None  # {(tail_entity_id, relation_id): [head_entity_ids, ...]}

        self.graph = None  # the pytorch geometric graph consisting of training triples, Data(x, edge_index, edge_attr);
        self.cluster_data = None  # generated subgraphs
        self.cluster_loader = None  # subgraph batch loader

        self.num_ori_rels = 0  # number of original relations

    def print_config(self):
        print("-----")
        print("### Running - `{}`".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print("#### Configurations")
        print("- load data from `{}`".format(self.data_path))
        if self.from_pre:
            print("- continue training based on: `{}`".format(self.model_path))
        else:
            print("- new training")
        print("- embedding dimension: `{}`".format(self.embed_dim))
        print("- number of negative triples: `{}`".format(self.neg_num))
        print("- learning rate: `{}`".format(self.lr))
        print("- dropout rate: `{}`".format(self.dropout))
        print("- margin: `{}`".format(self.margin))
        print("- number of bases: `{}`".format(self.num_bases))
        print("- rgcn aggregation scheme: `{}`".format(self.aggr))
        print("- number of subgraphs: `{}`".format(self.num_subgraphs))
        print("- cluster size: `{}`".format(self.cluster_size))
        print("- number of epochs: `{}`".format(self.num_epochs))
        print("- validation frequency: `{}`".format(self.valid_freq))
        print("- training batch size: `{}`".format(self.batch_size))
        print("- validation/test batch size: `{}`".format(self.vt_batch_size))
        print("- highest mrr: `{}`".format(self.highest_mrr))
        print("- device: `{}`".format(self.device))
        print("- evaluation device: `{}`".format(self.eval_device))
        if self.eval_sampling:
            print("- evaluation sampling size: `{}`".format(self.eval_sample_size))
        else:
            print("- use all entities as candidates in evaluation")

    def data_pre(self):
        print("#### Preparing Data")
        self.count, self.triples, self.hr2t, self.tr2h = read_data(self.data_path)
        print("- number of entities: `{}`".format(self.count["entity"]))
        self.num_ori_rels = self.count["relation"]
        print("- number of original relations: `{}`".format(self.count["relation"]))
        print("- number of original training triples: `{}`".format(self.count["train"]))
        print("- number of validation triples: `{}`".format(self.count["valid"]))
        print("- number of testing triples: `{}`".format(self.count["test"]))

        # create the training graph
        edge_index = torch.LongTensor(2, self.count["train"])  # triples' head and tail entity ids (changes after partitioning)
        edge_attr = torch.LongTensor(self.count["train"], 1)  # triples' relation ids (remains after partitioning)
        for triple_id in range(self.count["train"]):
            ids = self.triples["train"][triple_id, :]
            edge_index[0, triple_id] = ids[0]
            edge_attr[triple_id, 0] = ids[1]
            edge_index[1, triple_id] = ids[2]

        # add inverse relations
        sources = edge_index[0, :]
        targets = edge_index[1, :]
        edge_index = torch.cat((edge_index, torch.cat((targets.unsqueeze(0), sources.unsqueeze(0)), dim=0)), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr + self.count["relation"]), dim=0)
        self.count["relation"] = self.count["relation"] * 2

        # add self-loops
        self_loop_id = torch.LongTensor([self.count["relation"]])  # id of the self-loop relation
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index=edge_index, num_nodes=self.count["entity"])  # (2, num_train_triples * 2 + num_entities)
        edge_attr = torch.cat((edge_attr, self_loop_id.repeat(edge_index.size()[1] - edge_attr.size()[0], 1)), dim=0)  # (num_train_triples * 2 + num_entities, 1)
        self.count["relation"] += 1

        # construct a pytorch geometric data for the training graph
        x = torch.unsqueeze(torch.arange(self.count["entity"]), 1)  # use x to store original entity ids since entity ids in edge_index will change after partitioning
        self.graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # partition the training graph and instantiate the subgraph loader
        self.cluster_data = torch_geometric.data.ClusterData(data=self.graph, num_parts=self.num_subgraphs)
        self.cluster_loader = torch_geometric.data.ClusterLoader(cluster_data=self.cluster_data, batch_size=self.cluster_size, shuffle=True)

    def train(self):
        print("#### Model Training and Validation")

        # instantiate the model
        rgcn_lp = RgcnLP(num_entities=self.count["entity"], num_relations=self.count["relation"], dimension=self.embed_dim, num_bases=self.num_bases, aggr=self.aggr, dropout=self.dropout)
        if self.from_pre:
            rgcn_lp.load_state_dict(torch.load(self.model_path))
        rgcn_lp.to(self.device)

        # use Adam as the optimizer
        optimizer = torch.optim.Adam(params=rgcn_lp.parameters(), lr=self.lr)
        # use margin ranking loss as the loss function
        criterion = torch.nn.MarginRankingLoss(margin=self.margin)

        rgcn_lp.train()
        wandb.watch(rgcn_lp, criterion, log="all", log_freq=1)
        for epoch in range(self.num_epochs):
            print("* epoch {}".format(epoch))
            epoch_loss = 0.
            cluster_size = []
            for step, cluster in enumerate(self.cluster_loader):
                cluster_size.append(cluster.edge_index.size(1))

                # filter inverse and self-loop triples and sample negative triples
                pos_triples, neg_triples = train_triple_pre_all(ent_ids=cluster.x.squeeze(1),
                                                                head_ids=cluster.edge_index[0, :],
                                                                rel_ids=cluster.edge_attr.squeeze(1),
                                                                tail_ids=cluster.edge_index[1, :],
                                                                neg_num=self.neg_num,
                                                                self_rel_id=self.count["relation"] - 1)
                index_set = IndexSet(num_indices=pos_triples.size(0))
                index_loader = DataLoader(dataset=index_set, batch_size=self.batch_size, shuffle=True)

                for batch in index_loader:
                    pos_batch_triples = torch.index_select(input=pos_triples, index=batch, dim=0)
                    neg_batch_triples = torch.index_select(input=neg_triples, index=batch * self.neg_num, dim=0)
                    for i in range(self.neg_num):
                        if i > 0:
                            neg_batch_triples = torch.cat((neg_batch_triples, torch.index_select(input=neg_triples, index=batch * self.neg_num + i, dim=0)), dim=0)

                    optimizer.zero_grad()

                    # encode entities in the current batch
                    x = rgcn_lp.encode(ent_ids=cluster.x.squeeze(1).to(self.device),
                                       edge_index=cluster.edge_index.to(self.device),
                                       edge_type=cluster.edge_attr.squeeze(1).to(self.device))

                    train_triples = torch.cat((pos_batch_triples, neg_batch_triples), dim=0)

                    # compute scores for positive and negative triples
                    scores = rgcn_lp.decode(x=x, triples=train_triples.to(self.device))
                    pos_scores = scores[:pos_batch_triples.size(0)]  # (num_pos_triples)
                    neg_scores = torch.mean(torch.transpose(scores[pos_batch_triples.size(0):].view(self.neg_num, -1), 0, 1), dim=1)

                    # compute margin ranking loss
                    targets = torch.ones(pos_batch_triples.size(0))
                    batch_loss = criterion(input1=pos_scores, input2=neg_scores, target=targets.to(self.device))

                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss

            print("\t * number of triples in each cluster, min: {}, mean: {}, max: {}".format(min(cluster_size), 0 if len(cluster_size) == 0 else sum(cluster_size) / len(cluster_size), max(cluster_size)))
            print("\t * loss `{}`, time `{}`  ".format(epoch, epoch_loss, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if epoch % self.valid_freq == 0:
                self.evaluate(mode="valid", epoch=epoch, model=rgcn_lp)

            wandb.log({"epoch loss": epoch_loss}, step=epoch)

    def save(self):
        print("#### saving embeddings")

        with open(self.data_path + "id2ent.pickle", "rb") as f:
            id2ent = pickle.load(f)
        with open(self.data_path + "id2rel.pickle", "rb") as f:
            id2rel = pickle.load(f)

        tmp_model = RgcnLP(num_entities=self.count["entity"], num_relations=self.count["relation"], dimension=self.embed_dim, num_bases=self.num_bases, aggr=self.aggr, dropout=self.dropout)
        tmp_model.load_state_dict(torch.load(self.model_path))
        tmp_model.cpu()
        tmp_model.eval()
        with torch.no_grad():
            ent_embeds = tmp_model.encode(ent_ids=self.graph.x.squeeze(1), edge_index=self.graph.edge_index, edge_type=self.graph.edge_attr.squeeze(1))  # size: (num_entities, embed_dimension)
            rel_embeds = torch.index_select(input=tmp_model.distmult.rel_embeds, index=torch.LongTensor(range(self.num_ori_rels)), dim=0)  # size: (num_relations, embed_dimension)

        print("- saving entity embeddings")
        with open(self.data_path + "LSC-RGCN-embeds/entity_embs.csv", "w") as f:
            for ent_id in range(self.count["entity"]):
                ent_embed_line = id2ent[ent_id] + "," + str(ent_embeds[ent_id].tolist()).lstrip("[").rstrip("]").replace(" ", "")
                f.write(ent_embed_line + "\n")
        np.save(self.data_path + "LSC-RGCN-embeds/entity_embs.npy", ent_embeds.numpy())

        print("- saving relation embeddings")
        with open(self.data_path + "LSC-RGCN-embeds/relation_embs.csv", "w") as f:
            for rel_id in range(self.num_ori_rels):
                rel_embed_line = id2rel[rel_id] + "," + str(rel_embeds[rel_id].tolist()).lstrip("[").rstrip("]").replace(" ", "")
                f.write(rel_embed_line + "\n")
        np.save(self.data_path + "LSC-RGCN-embeds/relation_embs.npy", rel_embeds.numpy())

    def test(self):
        print("#### testing")
        test_model = RgcnLP(num_entities=self.count["entity"], num_relations=self.count["relation"], dimension=self.embed_dim, num_bases=self.num_bases, aggr=self.aggr, dropout=self.dropout)
        test_model.load_state_dict(torch.load(self.model_path))
        self.evaluate(mode="test", epoch=0, model=test_model)
        print("-----")
        print("  ")

    def evaluate(self, mode: str, epoch: int, model: RgcnLP):
        model.eval()
        with torch.no_grad():
            model.cpu()
            x = model.encode(ent_ids=self.graph.x.squeeze(1), edge_index=self.graph.edge_index, edge_type=self.graph.edge_attr.squeeze(1))
            x = x.to(self.eval_device)
            model.to(self.eval_device)
            all_head_ranks = None
            all_head_equals = None
            all_head_ranks2 = None
            all_tail_ranks = None
            all_tail_equals = None
            all_tail_ranks2 = None
            index_set = IndexSet(num_indices=self.count[mode])
            index_loader = DataLoader(dataset=index_set, batch_size=self.vt_batch_size, shuffle=False)
            for batch in index_loader:
                triples = torch.index_select(input=self.triples[mode], index=batch, dim=0).to(self.eval_device)  # size: (batch_size, 3)

                if self.eval_sampling:
                    sampled_entities = torch.LongTensor(list(torch.utils.data.RandomSampler(data_source=IndexSet(num_indices=self.count["entity"]), replacement=True,
                                                       num_samples=self.eval_sample_size)))  # sampled entities for evaluation, size: (eval_sample_size)
                    candidate_entities = sampled_entities.repeat(triples.size(0), 1).to(self.eval_device)  # size: (batch_size, eval_sample_size)
                else:
                    candidate_entities = torch.arange(self.count["entity"]).repeat(triples.size(0), 1).to(self.eval_device)  # size: (batch_size, num_entities)
                    self.eval_sample_size = self.count["entity"]

                # head prediction
                heads = triples[:, 0].view(-1, 1).to(self.eval_device)  # size: (batch_size, 1)
                test_heads = torch.cat((heads, candidate_entities), dim=1).unsqueeze(2)  # size: (batch_size, 1 + eval_sample_size, 1)

                no_heads = triples[:, 1:3].unsqueeze(1).repeat(1, 1 + self.eval_sample_size, 1).to(self.eval_device)  # (batch_size, 1 + eval_sample_size, 2)

                new_head_triples = torch.cat((test_heads, no_heads), dim=2).view(-1, 3).to(self.eval_device)  # size: (batch_size * (1 + eval_sample_size), 3)

                new_head_scores = model.decode(x=x, triples=new_head_triples)  # size: (batch_size * (1 + eval_sample_size))
                new_head_scores = new_head_scores.view(triples.size(0), 1 + self.eval_sample_size)  # size: (batch_size, (1 + eval_sample_size))
                correct_scores = new_head_scores[:, 0].unsqueeze(1)  # (batch_size, 1)
                if self.eval_device.type == "cuda":
                    false_positives = torch.nonzero(torch.cuda.BoolTensor(new_head_scores > correct_scores), as_tuple=True)[0]  # indices of random entities having higher scores than correct ones, size: (batch_size * num_false_positive_per_batch)
                    false_equals = torch.nonzero(torch.cuda.BoolTensor(new_head_scores == correct_scores), as_tuple=True)[0]
                else:
                    false_positives = torch.nonzero(torch.BoolTensor(new_head_scores > correct_scores), as_tuple=True)[0]  # indices of random entities having higher scores than correct ones
                    false_equals = torch.nonzero(torch.BoolTensor(new_head_scores == correct_scores), as_tuple=True)[0]
                false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.eval_device)), dim=0)
                head_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.eval_device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)
                head_equals = torch_scatter.scatter(src=torch.ones(false_equals.size(0)).to(torch.long).to(self.eval_device), index=false_equals, dim=0)
                head_equals = head_equals - 2

                if all_head_ranks is None:
                    all_head_ranks = head_ranks.to(torch.float)
                    all_head_equals = head_equals.to(torch.float)
                    all_head_ranks2 = (head_ranks + head_equals).to(torch.float)
                else:
                    all_head_ranks = torch.cat((all_head_ranks, head_ranks.to(torch.float)), dim=0)
                    all_head_equals = torch.cat((all_head_equals, head_equals.to(torch.float)), dim=0)
                    all_head_ranks2 = torch.cat((all_head_ranks2, (head_ranks + head_equals).to(torch.float)), dim=0)

                # tail prediction
                tails = triples[:, 2].view(-1, 1).to(self.eval_device)  # size: (batch_size, 1)
                test_tails = torch.cat((tails, candidate_entities), dim=1).unsqueeze(2)  # size: (batch_size, 1 + eval_sample_size, 1)

                no_tails = triples[:, 0:2].unsqueeze(1).repeat(1, 1 + self.eval_sample_size, 1).to(self.eval_device)  # size: (batch_size, 1 + eval_sample_size, 2)

                new_tail_triples = torch.cat((no_tails, test_tails), dim=2).view(-1, 3).to(self.eval_device)  # size: (batch_size * (1 + eval_sample_size)), 3)

                new_tail_scores = model.decode(x=x, triples=new_tail_triples)  # size: (batch_size * (1 + eval_sample_size)))
                new_tail_scores = new_tail_scores.view(triples.size(0), (1 + self.eval_sample_size))  # size: (batch_size, (1 + eval_sample_size))
                correct_scores = new_tail_scores[:, 0].unsqueeze(1)  # size: (batch_size, 1)
                if self.eval_device.type == "cuda":
                    false_positives = torch.nonzero(torch.cuda.BoolTensor(new_tail_scores > correct_scores), as_tuple=True)[0]  # indices of sampled entities having higher scores than correct ones
                    false_equals = torch.nonzero(torch.cuda.BoolTensor(new_tail_scores == correct_scores), as_tuple=True)[0]
                else:
                    false_positives = torch.nonzero(torch.BoolTensor(new_tail_scores > correct_scores), as_tuple=True)[0]  # indices of sampled entities having higher scores than correct ones
                    false_equals = torch.nonzero(torch.BoolTensor(new_tail_scores == correct_scores), as_tuple=True)[0]
                false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.eval_device)), dim=0)
                tail_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.eval_device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)
                tail_equals = torch_scatter.scatter(src=torch.ones(false_equals.size(0)).to(torch.long).to(self.eval_device), index=false_equals, dim=0)
                tail_equals = tail_equals - 2

                if all_tail_ranks is None:
                    all_tail_ranks = tail_ranks.to(torch.float)
                    all_tail_equals = tail_equals.to(torch.float)
                    all_tail_ranks2 = (tail_ranks + tail_equals).to(torch.float)
                else:
                    all_tail_ranks = torch.cat((all_tail_ranks, tail_ranks.to(torch.float)), dim=0)
                    all_tail_equals = torch.cat((all_tail_equals, tail_equals.to(torch.float)), dim=0)
                    all_tail_ranks2 = torch.cat((all_tail_ranks2, (tail_ranks + tail_equals).to(torch.float)), dim=0)

            h_mr = torch.mean(all_head_ranks)  # mean head rank
            h_mrr = torch.mean(1. / all_head_ranks)  # mean head reciprocal rank
            h_me = torch.mean(all_head_equals)  # number of candidate head entities having the same score
            h_mr2 = torch.mean(all_head_ranks2)  # mean head rank
            h_mrr2 = torch.mean(1. / all_head_ranks2)  # mean head reciprocal rank
            if self.eval_device.type == "cuda":
                h_hit1 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 1)).size(0) / all_head_ranks.size(0)  # head hit@1
                h_hit3 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 3)).size(0) / all_head_ranks.size(0)  # head hit@3
                h_hit10 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 10)).size(0) / all_head_ranks.size(0)  # head hit@10
            else:
                h_hit1 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 1)).size(0) / all_head_ranks.size(0)  # head hit@1
                h_hit3 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 3)).size(0) / all_head_ranks.size(0)  # head hit@3
                h_hit10 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 10)).size(0) / all_head_ranks.size(0)  # head hit@10

            t_mr = torch.mean(all_tail_ranks)  # mean tail rank
            t_mrr = torch.mean(1. / all_tail_ranks)  # mean tail reciprocal rank
            t_me = torch.mean(all_tail_equals)  # number of candidate tail entities having the same score
            t_mr2 = torch.mean(all_tail_ranks2)  # mean tail rank
            t_mrr2 = torch.mean(1. / all_tail_ranks2)  # mean tail reciprocal rank
            if self.eval_device.type == "cuda":
                t_hit1 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 1)).size(0) / all_tail_ranks.size(0)  # tail hit@1
                t_hit3 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 3)).size(0) / all_tail_ranks.size(0)  # tail hit@3
                t_hit10 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 10)).size(0) / all_tail_ranks.size(0)  # tail hit@10
            else:
                t_hit1 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 1)).size(0) / all_tail_ranks.size(0)  # tail hit@1
                t_hit3 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 3)).size(0) / all_tail_ranks.size(0)  # tail hit@3
                t_hit10 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 10)).size(0) / all_tail_ranks.size(0)  # tail hit@10

            print_dic = {"valid": "validation results (raw)", "test": "testing results (raw)"}
            if mode == "valid":
                print("\t * {}  at epoch `{}`".format(print_dic[mode], epoch))
            else:
                print("- {}  ".format(print_dic[mode]))
            print("   ")
            print("\t\t|  metric  |  head  |  tail  |  mean  |  ")
            print("\t\t|  ----  |  ----  |  ----  |  ----  |  ")
            print("\t\t|  mean reciprocal rank (MRR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mrr, t_mrr, (h_mrr + t_mrr)/2))
            print("\t\t|  mean rank (MR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mr, t_mr, (h_mr + t_mr)/2))
            print("\t\t|  mean equal (ME)  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_me, t_me, (h_me + t_me) / 2))
            print("\t\t|  mrr considering equals  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mrr2, t_mrr2, (h_mrr2 + t_mrr2) / 2))
            print("\t\t|  mr considering equals  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mr2, t_mr2, (h_mr2 + t_mr2) / 2))
            print("\t\t|  hits@1  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit1, t_hit1, (h_hit1 + t_hit1)/2))
            print("\t\t|  hits@3  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit3, t_hit3, (h_hit3 + t_hit3)/2))
            print("\t\t|  hits@10  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit10, t_hit10, (h_hit10 + t_hit10)/2))
            print("   ")

            if mode == "test":
                wandb.log({"epoch loss": 0.}, step=epoch)
            wandb.log({"MR": (h_mr + t_mr)/2,
                       "MRR": (h_mrr + t_mrr)/2,
                       "hits@1": (h_hit1 + t_hit1)/2,
                       "hits@3": (h_hit3 + t_hit3)/2,
                       "hits@10": (h_hit10 + t_hit10)/2,
                       "ME": (h_me + t_me) / 2,
                       "MR_ME": (h_mr2 + t_mr2) / 2,
                       "MRR_ME": (h_mrr2 + t_mrr2) / 2
                       }, step=epoch)

        if mode == "valid":
            if self.highest_mrr < (h_mrr + t_mrr)/2:
                self.highest_mrr = (h_mrr + t_mrr)/2
                torch.save(model.state_dict(), self.model_path)
                print("\t * model saved to `{}` at epoch `{}`   ".format(self.model_path, epoch))
        model.train()
        model.to(self.device)


if __name__ == "__main__":
    wandb.login()

    lrs = [0.001, 0.0005, 0.0001]
    batch_sizes = [64, 128]
    margins = [1, 5]
    params = list(product(lrs, batch_sizes, margins))

    for param in params:
        rgcn_main = RgcnMain(lr=param[0], batch_size=param[1], margin=param[2])
        config = {
            "data_path": rgcn_main.data_path,
            "model_path": rgcn_main.model_path,
            "from_pre": rgcn_main.from_pre,
            "num_epochs": rgcn_main.num_epochs,
            "valid_freq": rgcn_main.valid_freq,
            "learning_rate": rgcn_main.lr,
            "dropout": rgcn_main.dropout,
            "aggr": rgcn_main.aggr,
            "embed_dim": rgcn_main.embed_dim,
            "margin": rgcn_main.margin,
            "neg_num": rgcn_main.neg_num,
            "num_bases": rgcn_main.num_bases,
            "num_subgraphs": rgcn_main.num_subgraphs,
            "cluster_size": rgcn_main.cluster_size,
            "batch_size": rgcn_main.batch_size,
            "vt_batch_size": rgcn_main.vt_batch_size,
            "highest_mrr": rgcn_main.highest_mrr,
            "evaluation sampling": rgcn_main.eval_sampling,
            "sampling size": rgcn_main.eval_sample_size,
            "training_device": rgcn_main.device,
            "evaluation_device": rgcn_main.eval_device,
        }
        with wandb.init(entity="ruijie", project="rgcn", config=config, save_code=True, name="LR{}BS{}M{}".format(rgcn_main.lr, rgcn_main.batch_size, rgcn_main.margin)):
            rgcn_main.print_config()
            rgcn_main.data_pre()
            rgcn_main.train()
            rgcn_main.test()
            # rgcn_main.save()
            wandb.finish()
