import torch
import torch_scatter
from torch import Tensor
from datetime import datetime
from rgcn_models import RgcnLP
from torch.utils.data import DataLoader
from rgcn_utils import read_data, IndexSet, negative_sampling


class RgcnMain:
    def __init__(self):
        self.data_path = "../data/FB15K237/"
        self.model_path = "../pretrained/FB15K237/rgcn_lp.pt"

        self.from_pre = False  # True: continue training
        self.embed_dim = 200  # entity embedding dimension
        self.num_bases = 50  # bases of relation matrices
        self.aggr = "add"  # the aggregation scheme to use in RGCN
        self.batch_size = 10240  # train batch size
        self.vt_batch_size = 100  # validation/test batch size, please set it according to your memory size (current cost around 300GB)
        self.lr = 0.01  # learning rate
        self.num_epochs = 100  # number of epochs
        self.neg_num = 16  # number of negative triples for each positive triple
        self.valid_freq = 3  # validation frequency
        self.patience = 2  # determines when to early stop

        # self.count: {"entity": num_entities, "relation": num_relations, "train": num_train_triples, "valid": num_valid_triples, "test": num_test_triples};
        # self.triples: {"train": LongTensor(num_train_triples, 3), "valid": LongTensor(num_valid_triples, 3), "test": LongTensor(num_test_triples, 3)};
        # self.graph: the pytorch geometric graph consisting of training triples, Data(edge_index, edge_type);
        # self.hr2t: {(head_entity_id, relation_id): [tail_entity_ids, ...]}
        # self.tr2h: {(tail_entity_id, relation_id): [head_entity_ids, ...]}
        # self.correct_heads: {"valid": LongTensor(num_valid_triples, num_entities), "test": LongTensor(num_test_triples, num_entities)}
        # self.correct_tails: {"valid": LongTensor(num_valid_triples, num_entities), "test": LongTensor(num_test_triples, num_entities)}
        # note: inverse relations and self-loops have been added to the graph; self_loop_id = num_relations - 1;
        self.count, self.triples, self.graph, self.hr2t, self.tr2h, self.correct_heads, self.correct_tails = read_data(self.data_path)

        print("-----")
        print("### Running Time: `{}`".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print("#### configuration")
        print("- load data from `{}`".format(self.data_path))
        print("- continue training: `{}`".format(self.from_pre))
        print("- embedding dimension: `{}`".format(self.embed_dim))
        print("- number of bases: `{}`".format(self.num_bases))
        print("- rgcn aggregation scheme: `{}`".format(self.aggr))
        print("- train batch size: `{}`".format(self.batch_size))
        print("- validation/test batch size: `{}`".format(self.vt_batch_size))
        print("- learning rate: `{}`".format(self.lr))
        print("- number of epochs: `{}`".format(self.num_epochs))
        print("- number of negative triples: `{}`".format(self.neg_num))
        print("- number of entities: `{}`".format(self.count["entity"]))
        print("- number of relations: `{}`".format(self.count["relation"]))
        print("- number of training triples: `{}`".format(self.count["train"]))
        print("- number of validation triples: `{}`".format(self.count["valid"]))
        print("- number of testing triples: `{}`".format(self.count["test"]))
        print("- patience: `{}`".format(self.patience))

    # model training
    def main(self):
        # instantiate the link prediction model
        rgcn_lp = RgcnLP(in_dimension=self.embed_dim, out_dimension=self.embed_dim, num_entities=self.count["entity"], num_relations=self.count["relation"], num_bases=self.num_bases, aggr=self.aggr)
        if self.from_pre:
            rgcn_lp.load_state_dict(torch.load(self.model_path))

        # print parameter names of the model
        param_names = []
        for name, param in rgcn_lp.named_parameters():
            if param.requires_grad:
                param_names.append(name)
        print("- model parameters: `{}`".format(param_names))

        # use Adam as the optimizer
        optimizer = torch.optim.Adam(params=rgcn_lp.parameters(), lr=self.lr)
        # use binary cross entropy loss as the loss function
        criterion = torch.nn.BCELoss()

        # batch loaders for training/validation/testing triples
        train_index_set = IndexSet(num_indices=self.count["train"])
        train_index_loader = DataLoader(dataset=train_index_set, batch_size=self.batch_size, shuffle=True)
        valid_index_set = IndexSet(num_indices=self.count["valid"])
        valid_loader = DataLoader(dataset=valid_index_set, batch_size=self.vt_batch_size)
        test_index_set = IndexSet(num_indices=self.count["test"])
        test_loader = DataLoader(dataset=test_index_set, batch_size=self.vt_batch_size)

        train_triples = self.triples["train"]  # LongTensor, (num_train_triples, 3)
        highest_mrr = 0.
        print("#### training")
        patience = self.patience
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            # sample self.neg_num negative triples for each training triple
            neg_triples = negative_sampling(num_entities=self.count["entity"], num_triples=self.count["train"], neg_num=self.neg_num, train_triples=train_triples, hr2t=self.hr2t, tr2h=self.tr2h)  # (num_train_triples, neg_num, 3)
            for batch in train_index_loader:
                optimizer.zero_grad()
                batch_pos_triples = torch.index_select(input=train_triples, index=batch, dim=0)  # (current_batch_size, 3)
                pos_targets = torch.ones(batch.size()[0])  # (current_batch_size)
                batch_neg_triples = torch.index_select(input=neg_triples, index=batch, dim=0).view(-1, 3)  # (current_batch_size * neg_num, 3)
                neg_targets = torch.zeros(batch.size()[0] * self.neg_num)  # (current_batch_size * neg_num)
                batch_triples = torch.cat((batch_pos_triples, batch_neg_triples), dim=0)  # (current_batch_size + current_batch_size * neg_num, 3)
                targets = torch.cat((pos_targets, neg_targets), dim=0)  # (current_batch_size + current_batch_size * neg_num)
                scores = rgcn_lp(edge_index=self.graph.edge_index, edge_type=self.graph.edge_type, triple_batch=batch_triples)  # (current_batch_size + current_batch_size * neg_num)
                batch_loss = criterion(input=scores, target=targets)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            print("- epoch `{}`, loss `{}`, time `{}`  ".format(epoch, epoch_loss, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            if epoch % self.valid_freq == 0:
                current_mrr = self.v_and_t("valid", valid_loader, rgcn_lp, epoch)
                if highest_mrr < current_mrr:
                    patience = self.patience
                    highest_mrr = current_mrr
                    torch.save(rgcn_lp.state_dict(), self.model_path)
                    print("- model saved to `{}` at epoch `{}`   ".format(self.model_path, epoch))
                else:
                    patience -= 1
                    if patience == 0:
                        break
        print("#### testing")
        test_model = RgcnLP(in_dimension=self.embed_dim, out_dimension=self.embed_dim, num_entities=self.count["entity"], num_relations=self.count["relation"], num_bases=self.num_bases, aggr=self.aggr)
        test_model.load_state_dict(torch.load(self.model_path))
        self.v_and_t("test", test_loader, test_model, 0)
        print("-----")
        print("  ")

    # do validation/test
    # name: in ["valid", "test"]
    # index_loader: validation/test triple index loader
    # epoch: current epoch number
    def v_and_t(self, name: str, index_loader: DataLoader, model: RgcnLP, epoch: int) -> float:
        ranks = torch.zeros(5, 2)  # [[h_mr, t_mr], [h_mrr, t_mrr], [h_hit1, t_hit1], [h_hit3, t_hit3], [h_hit10, t_hit10]]
        batch_count = 0
        for batch in index_loader:
            # print("validation batch: {}, time: {}".format(batch_count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            triples = torch.index_select(input=self.triples[name], index=batch, dim=0)  # (vt_batch_size, 3)
            correct_heads = torch.index_select(input=self.correct_heads[name], index=batch, dim=0)  # (vt_batch_size, num_entities)
            correct_tails = torch.index_select(input=self.correct_tails[name], index=batch, dim=0)  # (vt_batch_size, num_entities)
            ranks += self.ranking(model, triples, correct_heads, correct_tails)
            batch_count += 1
        ranks = ranks / batch_count
        mean_ranks = torch.mean(ranks, dim=1)  # (5)

        print_dic = {"valid": "validation results", "test": "testing results"}
        if name == "valid":
            print("- {}  at epoch `{}`".format(print_dic[name], epoch))
        else:
            print("- {}  ".format(print_dic[name]))
        print("   ")
        print("|  metric  |  head  |  tail  |  mean  |  ")
        print("|  ----  |  ----  |  ----  |  ----  |  ")
        print("|  mean rank (MR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[0, 0], ranks[0, 1], mean_ranks[0]))
        print("|  mean reciprocal rank (MRR)  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[1, 0], ranks[1, 1], mean_ranks[1]))
        print("|  hit@1  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[2, 0], ranks[2, 1], mean_ranks[2]))
        print("|  hit@3  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[3, 0], ranks[3, 1], mean_ranks[3]))
        print("|  hit@10  |  `{}`  |  `{}`  |  `{}`  |  ".format(ranks[4, 0], ranks[4, 1], mean_ranks[4]))
        print("   ")
        return mean_ranks[1]  # MRR

    # compute the mean rank (MR), mean reciprocal rank (MRR) and hit@1, 3, 10 for the given triples
    # model: the link prediction model
    # triples: the triples for validation/test, (num_valid/test_triples, 3), num_valid/test_triples = vt_batch_size mostly
    # correct_heads, correct_tails: LongTensor(num_valid_triples, num_entities)
    def ranking(self, model: RgcnLP, triples: Tensor, correct_heads: Tensor, correct_tails: Tensor) -> Tensor:
        # candidate entities for each validation/testing triple
        entities = torch.arange(self.count["entity"]).repeat(triples.size()[0], 1).unsqueeze(2)  # (num_valid/test_triples, num_entities, 1)

        # for head prediction
        heads = triples[:, 0].view(-1, 1)  # (num_valid/test_triples, 1)
        no_heads = triples[:, 1:3].unsqueeze(1).repeat(1, self.count["entity"], 1)  # (num_valid/test_triples, num_entities, 2)
        new_head_triples = torch.cat((entities, no_heads), dim=2).view(-1, 3)  # (num_valid/test_triples * num_entities, 3)

        # for tail prediction
        tails = triples[:, 2].view(-1, 1)  # (num_valid/test_triples, 1)
        no_tails = triples[:, 0:2].unsqueeze(1).repeat(1, self.count["entity"], 1)  # (num_valid/test_triples, num_entities, 2)
        new_tail_triples = torch.cat((no_tails, entities), dim=2).view(-1, 3)  # (num_valid/test_triples * num_entities, 3)

        # evaluate the model
        model.eval()
        with torch.no_grad():
            new_head_scores = model(edge_index=self.graph.edge_index, edge_type=self.graph.edge_type, triple_batch=new_head_triples)  # (num_valid/test_triples * num_entities)
            new_head_scores = new_head_scores.view(triples.size()[0], self.count["entity"])  # (num_valid/test_triples, num_entities)
            filtered_head_scores = torch.gather(input=new_head_scores, dim=1, index=correct_heads)  # (num_valid/test_triples, num_entities)
            correct_scores = torch.gather(input=filtered_head_scores, dim=1, index=heads)  # (num_valid/test_triples, 1)
            false_positives = torch.nonzero(torch.BoolTensor(filtered_head_scores > correct_scores), as_tuple=True)  # indices of the entities having higher scores than correct ones
            head_ranks = torch_scatter.scatter(src=torch.ones(false_positives[0].size()[0]).to(torch.long), index=false_positives[0], dim=0)  # number of false positives for each valid/test triple, (num_valid/test_triples)
            head_ranks = head_ranks + 1  # ranks of correct head entities

            h_mr = torch.mean(head_ranks.to(torch.float))  # mean head rank
            h_mrr = torch.mean(1. / head_ranks.to(torch.float))  # mean head reciprocal rank
            h_hit1 = torch.nonzero(torch.BoolTensor(head_ranks <= 1)).size()[0] / head_ranks.size()[0]  # head hit@1
            h_hit3 = torch.nonzero(torch.BoolTensor(head_ranks <= 3)).size()[0] / head_ranks.size()[0]  # head hit@3
            h_hit10 = torch.nonzero(torch.BoolTensor(head_ranks <= 10)).size()[0] / head_ranks.size()[0]  # head hit@10

            new_tail_scores = model(edge_index=self.graph.edge_index, edge_type=self.graph.edge_type, triple_batch=new_tail_triples)  # (num_valid/test_triples * num_entities)
            new_tail_scores = new_tail_scores.view(triples.size()[0], self.count["entity"])  # (num_valid/test_triples, num_entities)
            filtered_tail_scores = torch.gather(input=new_tail_scores, dim=1, index=correct_tails)  # (num_valid/test_triples, num_entities)
            correct_scores = torch.gather(input=filtered_tail_scores, dim=1, index=tails)  # (num_valid/test_triples, 1)
            false_positives = torch.nonzero(torch.BoolTensor(filtered_tail_scores > correct_scores), as_tuple=True)  # indices of the entities having higher scores than correct ones
            tail_ranks = torch_scatter.scatter(src=torch.ones(false_positives[0].size()[0]).to(torch.long), index=false_positives[0], dim=0)  # number of false positives for each valid/test triple, (num_valid/test_triples)
            tail_ranks = tail_ranks + 1  # ranks of correct tail entities

            t_mr = torch.mean(tail_ranks.to(torch.float))  # mean tail rank
            t_mrr = torch.mean(1. / tail_ranks.to(torch.float))  # mean tail reciprocal rank
            t_hit1 = torch.nonzero(torch.BoolTensor(tail_ranks <= 1)).size()[0] / tail_ranks.size()[0]  # tail hit@1
            t_hit3 = torch.nonzero(torch.BoolTensor(tail_ranks <= 3)).size()[0] / tail_ranks.size()[0]  # tail hit@3
            t_hit10 = torch.nonzero(torch.BoolTensor(tail_ranks <= 10)).size()[0] / tail_ranks.size()[0]  # tail hit@10

        model.train()
        return torch.FloatTensor([[h_mr, t_mr], [h_mrr, t_mrr], [h_hit1, t_hit1], [h_hit3, t_hit3], [h_hit10, t_hit10]])


# do link prediction
if __name__ == "__main__":
    rgcn_main = RgcnMain()
    rgcn_main.main()

