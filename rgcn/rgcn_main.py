import torch
import torch_scatter
import torch_geometric
from datetime import datetime
from rgcn_models import RgcnLP
from torch.utils.data import DataLoader
from rgcn_utils import read_data, IndexSet, train_triple_pre


class RgcnMain:
    def __init__(self):
        self.data_path = "../data/FB15K237/"
        self.model_path = "../pretrained/FB15K237/rgcn_lp.pt"

        self.from_pre = False  # True: continue training
        self.num_epochs = 1000  # number of training epochs
        self.valid_freq = 1  # do validation every x training epochs
        self.lr = 0.0005  # learning rate
        self.dropout = 0.2  # dropout rate

        self.aggr = "add"  # aggregation scheme to use in RGCN, "add" | "mean" | "max"
        self.embed_dim = 100  # entity embedding dimension

        self.neg_num = 1  # number of negative triples for each positive triple
        self.num_bases = 50  # number of bases for relation matrices in RGCN

        self.num_subgraphs = 200  # partition the training graph into x subgraphs; please set it according to your GPU memory (if available)
        self.subgraph_batch_size = 24  # number of subgraphs in each batch
        self.vt_batch_size = 12  # validation/test batch size

        self.gpu = "cuda:2"  # the device to use, "cpu" | "cuda:x"
        if torch.cuda.is_available():
            self.device = torch.device(self.gpu)
        else:
            self.device = torch.device("cpu")

        self.count = None  # {"entity": num_entities, "relation": num_relations, "train": num_train_triples, "valid": num_valid_triples, "test": num_test_triples};
        self.triples = None  # {"train": LongTensor(num_train_triples, 3), "valid": LongTensor(num_valid_triples, 3), "test": LongTensor(num_test_triples, 3)};
        self.hr2t = None  # {(head_entity_id, relation_id): [tail_entity_ids, ...]}
        self.tr2h = None  # {(tail_entity_id, relation_id): [head_entity_ids, ...]}
        self.correct_heads = None  # {"valid": LongTensor(num_valid_triples, num_entities), "test": LongTensor(num_test_triples, num_entities)}
        self.correct_tails = None  # {"valid": LongTensor(num_valid_triples, num_entities), "test": LongTensor(num_test_triples, num_entities)}

        self.graph = None  # the pytorch geometric graph consisting of training triples, Data(x, edge_index, edge_attr);
        self.cluster_data = None  # generated subgraphs
        self.cluster_loader = None  # subgraph batch loader
        self.highest_mrr = 0.  # highest mrr in validation

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
        print("- number of bases: `{}`".format(self.num_bases))
        print("- rgcn aggregation scheme: `{}`".format(self.aggr))
        print("- number of subgraphs: {}".format(self.num_subgraphs))
        print("- training subgraph batch size: `{}`".format(self.subgraph_batch_size))
        print("- number of epochs: `{}`".format(self.num_epochs))
        print("- validation frequency: `{}`".format(self.valid_freq))
        print("- validation/test triple batch size: `{}`".format(self.vt_batch_size))
        print("- device: `{}`".format(self.device))

    def data_pre(self):
        print("#### Preparing Data")
        self.count, self.triples, self.hr2t, self.tr2h, self.correct_heads, self.correct_tails = read_data(self.data_path)
        print("- number of entities: `{}`".format(self.count["entity"]))
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
        self.cluster_loader = torch_geometric.data.ClusterLoader(cluster_data=self.cluster_data, batch_size=self.subgraph_batch_size, shuffle=True)

    def train(self):
        print("#### Model Training and Validation")

        # instantiate the model
        rgcn_lp = RgcnLP(num_entities=self.count["entity"], num_relations=self.count["relation"], dimension=self.embed_dim, num_bases=self.num_bases, aggr=self.aggr, dropout=self.dropout)
        if self.from_pre:
            rgcn_lp.load_state_dict(torch.load(self.model_path))
        rgcn_lp.to(self.device)

        # use Adam as the optimizer
        optimizer = torch.optim.Adam(params=rgcn_lp.parameters(), lr=self.lr)
        # use binary cross entropy loss as the loss function
        criterion = torch.nn.BCELoss()

        rgcn_lp.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            for step, cluster in enumerate(self.cluster_loader):
                optimizer.zero_grad()

                # encode entities in the current batch
                x = rgcn_lp.encode(ent_ids=cluster.x.squeeze(1).to(self.device),
                                   edge_index=cluster.edge_index.to(self.device),
                                   edge_type=cluster.edge_attr.squeeze(1).to(self.device))

                # filter inverse and self-loop triples and sample negative triples
                pos_triples, neg_triples = train_triple_pre(ent_ids=cluster.x.squeeze(1),
                                                            head_ids=cluster.edge_index[0, :],
                                                            rel_ids=cluster.edge_attr.squeeze(1),
                                                            tail_ids=cluster.edge_index[1, :],
                                                            hr2t=self.hr2t, tr2h=self.tr2h, neg_num=self.neg_num)
                train_triples = torch.cat((pos_triples, neg_triples), dim=0)

                # compute scores for positive and negative triples
                scores = rgcn_lp.decode(x=x, triples=train_triples.to(self.device))

                # compute binary cross-entropy loss
                pos_targets = torch.ones(pos_triples.size(0))
                neg_targets = torch.zeros(neg_triples.size(0))
                train_targets = torch.cat((pos_targets, neg_targets), dim=0)
                batch_loss = criterion(input=scores, target=train_targets.to(self.device))

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            print("- epoch `{}`, loss `{}`, time `{}`  ".format(epoch, epoch_loss, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if epoch % self.valid_freq == 0:
                self.evaluate(mode="valid", epoch=epoch, model=rgcn_lp)

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
            x = x.to(self.device)
            model.to(self.device)
            ranks = torch.zeros(5, 2).to(self.device)  # [[h_mr, t_mr], [h_mrr, t_mrr], [h_hit1, t_hit1], [h_hit3, t_hit3], [h_hit10, t_hit10]]
            batch_num = 0
            index_set = IndexSet(num_indices=self.count[mode])
            index_loader = DataLoader(dataset=index_set, batch_size=self.vt_batch_size, shuffle=False)
            for batch in index_loader:
                triples = torch.index_select(input=self.triples[mode], index=batch, dim=0).to(self.device)  # (batch_size, 3)
                correct_heads = torch.index_select(input=self.correct_heads[mode], index=batch, dim=0).to(self.device)  # (batch_size, num_entities)
                correct_tails = torch.index_select(input=self.correct_tails[mode], index=batch, dim=0).to(self.device)  # (batch_size, num_entities)
                all_entities = torch.arange(self.count["entity"]).repeat(triples.size()[0], 1).unsqueeze(2).to(self.device)  # (batch_size, num_entities, 1)

                # head prediction
                heads = triples[:, 0].view(-1, 1).to(self.device)  # (batch_size, 1)
                no_heads = triples[:, 1:3].unsqueeze(1).repeat(1, self.count["entity"], 1).to(self.device)  # (batch_size, num_entities, 2)
                new_head_triples = torch.cat((all_entities, no_heads), dim=2).view(-1, 3).to(self.device)  # (batch_size * num_entities, 3)

                new_head_scores = model.decode(x=x, triples=new_head_triples)  # (batch_size * num_entities)
                new_head_scores = new_head_scores.view(triples.size()[0], self.count["entity"])  # (batch_size, num_entities)
                filtered_head_scores = torch.gather(input=new_head_scores, dim=1, index=correct_heads)  # (batch_size, num_entities)
                correct_scores = torch.gather(input=filtered_head_scores, dim=1, index=heads)  # (batch_size, 1)
                if self.device.type == "cuda":
                    false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_head_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                else:
                    false_positives = torch.nonzero(torch.BoolTensor(filtered_head_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)
                head_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)

                h_mr = torch.mean(head_ranks.to(torch.float))  # mean head rank
                h_mrr = torch.mean(1. / head_ranks.to(torch.float))  # mean head reciprocal rank
                if self.device.type == "cuda":
                    h_hit1 = torch.nonzero(torch.cuda.BoolTensor(head_ranks <= 1)).size()[0] / head_ranks.size()[0]  # head hit@1
                    h_hit3 = torch.nonzero(torch.cuda.BoolTensor(head_ranks <= 3)).size()[0] / head_ranks.size()[0]  # head hit@3
                    h_hit10 = torch.nonzero(torch.cuda.BoolTensor(head_ranks <= 10)).size()[0] / head_ranks.size()[0]  # head hit@10
                else:
                    h_hit1 = torch.nonzero(torch.BoolTensor(head_ranks <= 1)).size()[0] / head_ranks.size()[0]  # head hit@1
                    h_hit3 = torch.nonzero(torch.BoolTensor(head_ranks <= 3)).size()[0] / head_ranks.size()[0]  # head hit@3
                    h_hit10 = torch.nonzero(torch.BoolTensor(head_ranks <= 10)).size()[0] / head_ranks.size()[0]  # head hit@10

                # tail prediction
                tails = triples[:, 2].view(-1, 1).to(self.device)  # (batch_size, 1)
                no_tails = triples[:, 0:2].unsqueeze(1).repeat(1, self.count["entity"], 1).to(self.device)  # (batch_size, num_entities, 2)
                new_tail_triples = torch.cat((no_tails, all_entities), dim=2).view(-1, 3).to(self.device)  # (batch_size * num_entities, 3)

                new_tail_scores = model.decode(x=x, triples=new_tail_triples)  # (batch_size * num_entities)
                new_tail_scores = new_tail_scores.view(triples.size()[0], self.count["entity"])  # (batch_size, num_entities)
                filtered_tail_scores = torch.gather(input=new_tail_scores, dim=1, index=correct_tails)  # (batch_size, num_entities)
                correct_scores = torch.gather(input=filtered_tail_scores, dim=1, index=tails)  # (batch_size, 1)
                if self.device.type == "cuda":
                    false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_tail_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                else:
                    false_positives = torch.nonzero(torch.BoolTensor(filtered_tail_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)
                tail_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)

                t_mr = torch.mean(tail_ranks.to(torch.float))  # mean tail rank
                t_mrr = torch.mean(1. / tail_ranks.to(torch.float))  # mean tail reciprocal rank
                if self.device.type == "cuda":
                    t_hit1 = torch.nonzero(torch.cuda.BoolTensor(tail_ranks <= 1)).size()[0] / tail_ranks.size()[0]  # tail hit@1
                    t_hit3 = torch.nonzero(torch.cuda.BoolTensor(tail_ranks <= 3)).size()[0] / tail_ranks.size()[0]  # tail hit@3
                    t_hit10 = torch.nonzero(torch.cuda.BoolTensor(tail_ranks <= 10)).size()[0] / tail_ranks.size()[0]  # tail hit@10
                else:
                    t_hit1 = torch.nonzero(torch.BoolTensor(tail_ranks <= 1)).size()[0] / tail_ranks.size()[0]  # tail hit@1
                    t_hit3 = torch.nonzero(torch.BoolTensor(tail_ranks <= 3)).size()[0] / tail_ranks.size()[0]  # tail hit@3
                    t_hit10 = torch.nonzero(torch.BoolTensor(tail_ranks <= 10)).size()[0] / tail_ranks.size()[0]  # tail hit@10

                ranks += torch.FloatTensor([[h_mr, t_mr], [h_mrr, t_mrr], [h_hit1, t_hit1], [h_hit3, t_hit3], [h_hit10, t_hit10]]).to(self.device)
                batch_num += 1
            ranks = ranks / batch_num
            mean_ranks = torch.mean(ranks, dim=1)  # (5)
            print_dic = {"valid": "validation results", "test": "testing results"}
            if mode == "valid":
                print("- {}  at epoch `{}`".format(print_dic[mode], epoch))
            else:
                print("- {}  ".format(print_dic[mode]))
            print("   ")
            print("|  metric  |  head  |  tail  |  mean  |  ")
            print("|  ----  |  ----  |  ----  |  ----  |  ")
            print("|  mean reciprocal rank (MRR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[1, 0], ranks[1, 1], mean_ranks[1]))
            print("|  mean rank (MR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[0, 0], ranks[0, 1], mean_ranks[0]))
            print("|  hits@1  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[2, 0], ranks[2, 1], mean_ranks[2]))
            print("|  hits@3  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[3, 0], ranks[3, 1], mean_ranks[3]))
            print("|  hits@10  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[4, 0], ranks[4, 1], mean_ranks[4]))
            print("   ")
        if mode == "valid":
            if self.highest_mrr < mean_ranks[1]:
                self.highest_mrr = mean_ranks[1]
                torch.save(model.state_dict(), self.model_path)
                print("- model saved to `{}` at epoch `{}`   ".format(self.model_path, epoch))
        model.train()


if __name__ == "__main__":
    rgcn_main = RgcnMain()
    rgcn_main.data_pre()
    rgcn_main.train()
    rgcn_main.test()
