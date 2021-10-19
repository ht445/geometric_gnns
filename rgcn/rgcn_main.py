import torch
import torchviz
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
        self.num_epochs = 50  # number of training epochs
        self.valid_freq = 1  # do validation every x training epochs
        self.lr = 0.0005  # learning rate
        self.dropout = 0.2  # dropout rate
        self.penalty = 0.01  # regularization loss ratio

        self.aggr = "add"  # aggregation scheme to use in RGCN, "add" | "mean" | "max"
        self.embed_dim = 100  # entity embedding dimension

        self.neg_num = 1  # number of negative triples for each positive triple
        self.num_bases = 50  # number of bases for relation matrices in RGCN

        self.num_subgraphs = 200  # partition the training graph into x subgraphs; please set it according to your GPU memory (if available)
        self.cluster_size = 24  # number of subgraphs in each cluster

        self.batch_size = 256  # training batch size
        self.vt_batch_size = 12  # validation/test batch size

        self.highest_mrr = 0.  # highest mrr in validation

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
        print("- regularization loss ratio: `{}`".format(self.penalty))
        print("- number of bases: `{}`".format(self.num_bases))
        print("- rgcn aggregation scheme: `{}`".format(self.aggr))
        print("- number of subgraphs: `{}`".format(self.num_subgraphs))
        print("- training cluster size: `{}`".format(self.cluster_size))
        print("- training batch size: `{}`".format(self.batch_size))
        print("- number of epochs: `{}`".format(self.num_epochs))
        print("- validation frequency: `{}`".format(self.valid_freq))
        print("- validation/test triple batch size: `{}`".format(self.vt_batch_size))
        print("- highest mrr: `{}`".format(self.highest_mrr))
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
        # use binary cross entropy loss as the loss function
        criterion = torch.nn.BCELoss()

        rgcn_lp.train()
        plot = True
        for epoch in range(self.num_epochs):
            print("* epoch {}".format(epoch))
            epoch_loss = 0.
            cluster_size = []
            for step, cluster in enumerate(self.cluster_loader):
                cluster_size.append(cluster.edge_index.size(1))

                # filter inverse and self-loop triples and sample negative triples
                pos_triples, neg_triples = train_triple_pre(ent_ids=cluster.x.squeeze(1),
                                                            head_ids=cluster.edge_index[0, :],
                                                            rel_ids=cluster.edge_attr.squeeze(1),
                                                            tail_ids=cluster.edge_index[1, :],
                                                            hr2t=self.hr2t, tr2h=self.tr2h, neg_num=self.neg_num)

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

                    # compute scores for positive and negative triples
                    train_triples = torch.cat((pos_batch_triples, neg_batch_triples), dim=0)
                    scores = rgcn_lp.decode(x=x, triples=train_triples.to(self.device))

                    # compute binary cross-entropy loss
                    pos_targets = torch.ones(pos_batch_triples.size(0))
                    neg_targets = torch.zeros(neg_batch_triples.size(0))
                    train_targets = torch.cat((pos_targets, neg_targets), dim=0)
                    bce_loss = criterion(input=scores, target=train_targets.to(self.device))

                    if plot:
                        dot = torchviz.make_dot(bce_loss, params=dict(rgcn_lp.named_parameters()))
                        dot.format = 'png'
                        dot.render('./compgcn_lp_graph')
                        plot = False

                    # compute regularization loss
                    reg_loss = rgcn_lp.reg_loss(x=x, rel_ids=train_triples[:, 1].to(self.device))

                    batch_loss = bce_loss + self.penalty * reg_loss
                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss

            print("\t * number of triples in each cluster, min: {}, mean: {}, max: {}".format(min(cluster_size), 0 if len(cluster_size) == 0 else sum(cluster_size) / len(cluster_size), max(cluster_size)))
            print("\t * loss `{}`, time `{}`  ".format(epoch_loss, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

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
            all_head_ranks = None
            all_tail_ranks = None
            index_set = IndexSet(num_indices=self.count[mode])
            index_loader = DataLoader(dataset=index_set, batch_size=self.vt_batch_size, shuffle=False)
            for batch in index_loader:
                triples = torch.index_select(input=self.triples[mode], index=batch, dim=0).to(self.device)  # (batch_size, 3)

                all_entities = torch.arange(self.count["entity"]).repeat(triples.size()[0], 1).unsqueeze(2).to(self.device)  # (batch_size, num_entities, 1)

                # head prediction
                heads = triples[:, 0].view(-1, 1).to(self.device)  # (batch_size, 1)
                no_heads = triples[:, 1:3].unsqueeze(1).repeat(1, self.count["entity"], 1).to(self.device)  # (batch_size, num_entities, 2)
                new_head_triples = torch.cat((all_entities, no_heads), dim=2).view(-1, 3).to(self.device)  # (batch_size * num_entities, 3)

                new_head_scores = model.decode(x=x, triples=new_head_triples)  # (batch_size * num_entities)
                new_head_scores = new_head_scores.view(triples.size()[0], self.count["entity"])  # (batch_size, num_entities)

                if mode == "valid":
                    filtered_head_scores = new_head_scores
                elif mode == "test":
                    correct_heads = torch.index_select(input=self.correct_heads[mode], index=batch, dim=0).to(self.device)  # (batch_size, num_entities)
                    filtered_head_scores = torch.gather(input=new_head_scores, dim=1, index=correct_heads)  # (batch_size, num_entities)
                correct_scores = torch.gather(input=filtered_head_scores, dim=1, index=heads)  # (batch_size, 1)

                if self.device.type == "cuda":
                    if mode == "valid":
                        false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_head_scores >= correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                    elif mode == "test":
                        false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_head_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                        false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)
                else:
                    if mode == "valid":
                        false_positives = torch.nonzero(torch.BoolTensor(filtered_head_scores >= correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                    elif mode == "test":
                        false_positives = torch.nonzero(torch.BoolTensor(filtered_head_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                        false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)

                head_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)
                if all_head_ranks is None:
                    all_head_ranks = head_ranks.to(torch.float)
                else:
                    all_head_ranks = torch.cat((all_head_ranks, head_ranks.to(torch.float)), dim=0)

                # tail prediction
                tails = triples[:, 2].view(-1, 1).to(self.device)  # (batch_size, 1)
                no_tails = triples[:, 0:2].unsqueeze(1).repeat(1, self.count["entity"], 1).to(self.device)  # (batch_size, num_entities, 2)
                new_tail_triples = torch.cat((no_tails, all_entities), dim=2).view(-1, 3).to(self.device)  # (batch_size * num_entities, 3)

                new_tail_scores = model.decode(x=x, triples=new_tail_triples)  # (batch_size * num_entities)
                new_tail_scores = new_tail_scores.view(triples.size()[0], self.count["entity"])  # (batch_size, num_entities)

                if mode == "valid":
                    filtered_tail_scores = new_tail_scores
                elif mode == "test":
                    correct_tails = torch.index_select(input=self.correct_tails[mode], index=batch, dim=0).to(self.device)  # (batch_size, num_entities)
                    filtered_tail_scores = torch.gather(input=new_tail_scores, dim=1, index=correct_tails)  # (batch_size, num_entities)
                correct_scores = torch.gather(input=filtered_tail_scores, dim=1, index=tails)  # (batch_size, 1)

                if self.device.type == "cuda":
                    if mode == "valid":
                        false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_tail_scores >= correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                    elif mode == "test":
                        false_positives = torch.nonzero(torch.cuda.BoolTensor(filtered_tail_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                        false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)
                else:
                    if mode == "valid":
                        false_positives = torch.nonzero(torch.BoolTensor(filtered_tail_scores >= correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                    elif mode == "test":
                        false_positives = torch.nonzero(torch.BoolTensor(filtered_tail_scores > correct_scores), as_tuple=True)[0]  # indices of the entities having higher scores than correct ones
                        false_positives = torch.cat((false_positives, torch.arange(correct_scores.size(0)).to(self.device)), dim=0)

                tail_ranks = torch_scatter.scatter(src=torch.ones(false_positives.size(0)).to(torch.long).to(self.device), index=false_positives, dim=0)  # number of false positives for each valid/test triple, (batch_size)
                if all_tail_ranks is None:
                    all_tail_ranks = tail_ranks.to(torch.float)
                else:
                    all_tail_ranks = torch.cat((all_tail_ranks, tail_ranks.to(torch.float)), dim=0)

            h_mr = torch.mean(all_head_ranks)  # mean head rank
            h_mrr = torch.mean(1. / all_head_ranks)  # mean head reciprocal rank
            if self.device.type == "cuda":
                h_hit1 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 1)).size(0) / all_head_ranks.size(0)  # head hit@1
                h_hit3 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 3)).size(0) / all_head_ranks.size(0)  # head hit@3
                h_hit10 = torch.nonzero(torch.cuda.BoolTensor(all_head_ranks <= 10)).size(0) / all_head_ranks.size(0)  # head hit@10
            else:
                h_hit1 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 1)).size(0) / all_head_ranks.size(0)  # head hit@1
                h_hit3 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 3)).size(0) / all_head_ranks.size(0)  # head hit@3
                h_hit10 = torch.nonzero(torch.BoolTensor(all_head_ranks <= 10)).size(0) / all_head_ranks.size(0)  # head hit@10

            t_mr = torch.mean(all_tail_ranks)  # mean tail rank
            t_mrr = torch.mean(1. / all_tail_ranks)  # mean tail reciprocal rank
            if self.device.type == "cuda":
                t_hit1 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 1)).size(0) / all_tail_ranks.size(0)  # tail hit@1
                t_hit3 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 3)).size(0) / all_tail_ranks.size(0)  # tail hit@3
                t_hit10 = torch.nonzero(torch.cuda.BoolTensor(all_tail_ranks <= 10)).size(0) / all_tail_ranks.size(0)  # tail hit@10
            else:
                t_hit1 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 1)).size(0) / all_tail_ranks.size(0)  # tail hit@1
                t_hit3 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 3)).size(0) / all_tail_ranks.size(0)  # tail hit@3
                t_hit10 = torch.nonzero(torch.BoolTensor(all_tail_ranks <= 10)).size(0) / all_tail_ranks.size(0)  # tail hit@10

            print_dic = {"valid": "validation results (raw)", "test": "testing results (filtered)"}
            if mode == "valid":
                print("\t * {}  at epoch `{}`".format(print_dic[mode], epoch))
            else:
                print("- {}  ".format(print_dic[mode]))
            print("   ")
            print("\t\t|  metric  |  head  |  tail  |  mean  |  ")
            print("\t\t|  ----  |  ----  |  ----  |  ----  |  ")
            print("\t\t|  mean reciprocal rank (MRR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mrr, t_mrr, (h_mrr + t_mrr)/2))
            print("\t\t|  mean rank (MR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_mr, t_mr, (h_mr + t_mr)/2))
            print("\t\t|  hits@1  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit1, t_hit1, (h_hit1 + t_hit1)/2))
            print("\t\t|  hits@3  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit3, t_hit3, (h_hit3 + t_hit3)/2))
            print("\t\t|  hits@10  |  `{}`  |  `{}`  |  `{}`  |  ".format(h_hit10, t_hit10, (h_hit10 + t_hit10)/2))
            print("   ")
        if mode == "valid":
            if self.highest_mrr < (h_mrr + t_mrr)/2:
                self.highest_mrr = (h_mrr + t_mrr)/2
                torch.save(model.state_dict(), self.model_path)
                print("\t * model saved to `{}` at epoch `{}`   ".format(self.model_path, epoch))
        model.train()


if __name__ == "__main__":
    rgcn_main = RgcnMain()
    rgcn_main.data_pre()
    rgcn_main.train()
    rgcn_main.test()
