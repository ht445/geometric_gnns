import torch
from rgcn_models import RgcnLP
from datetime import datetime
from torch.utils.data import DataLoader
from rgcn_utils import read_data, IndexSet, negative_sampling


class RgcnMain:
    def __init__(self):
        self.data_path = "../data/FB15K237/"
        self.model_path = "../pretrained/FB15K237/rgcn_lp.pt"

        self.from_pre = False  # True: continue training
        self.embed_dim = 100  # entity embedding dimension
        self.num_bases = 24  # bases of relation matrices
        self.aggr = "add"  # the aggregation scheme to use in RGCN
        self.batch_size = 20480  # batch size
        self.lr = 0.01  # learning rate
        self.num_epochs = 100  # number of epochs
        self.neg_num = 5  # number of negative triples for each positive triple

        # self.count: {"entity": num_entities, "relation": num_relations, "train": num_train_triples, "valid": num_valid_triples, "test": num_test_triples};
        # self.triples: {"train": LongTensor(num_train_triples, 3), "valid": LongTensor(num_valid_triples, 3), "test": LongTensor(num_test_triples, 3)};
        # self.graph: the pytorch geometric graph consisting of training triples, Data(edge_index, edge_type);
        # note: self-loops have been added to the graph; self_loop_id = num_relations - 1;
        self.count, self.triples, self.graph = read_data(self.data_path)

        print("-- configuration -- ")
        print("continue training: {}".format(self.from_pre))
        print("embedding dimension: {}".format(self.embed_dim))
        print("number of bases: {}".format(self.num_bases))
        print("rgcn aggregation scheme: {}".format(self.aggr))
        print("batch size: {}".format(self.batch_size))
        print("learning rate: {}".format(self.lr))
        print("number of epochs: {}".format(self.num_epochs))
        print("number of negative triples: {}".format(self.neg_num))

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
        print("model parameters: {}".format(param_names))

        # use Adam as the optimizer
        optimizer = torch.optim.Adam(params=rgcn_lp.parameters(), lr=self.lr)
        # use binary cross entropy loss as the loss function
        criterion = torch.nn.BCELoss()

        # batch loader for training triples
        index_set = IndexSet(num_indices=self.count["train"])
        index_loader = DataLoader(dataset=index_set, batch_size=self.batch_size, shuffle=True)

        train_triples = self.triples["train"]  # LongTensor, (num_train_triples, 3)
        lowest_loss = 100000.
        print("-- training --")
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            # sample self.neg_num negative triples for each training triple
            neg_triples = negative_sampling(num_entities=self.count["entity"], num_triples=self.count["train"], neg_num=self.neg_num, train_triples=train_triples)  # (num_train_triples, neg_num, 3)
            for batch in index_loader:
                optimizer.zero_grad()
                batch_pos_triples = torch.index_select(input=train_triples, index=batch, dim=0)  # (current_batch_size, 3)
                pos_targets = torch.ones(batch.size()[0])  # (current_batch_size)
                batch_neg_triples = torch.index_select(input=neg_triples, index=batch, dim=0).view(-1, 3)  # (current_batch_size * neg_num, 3)
                neg_targets = torch.zeros(batch.size()[0] * self.neg_num)  # (current_batch_size * neg_num)
                batch_triples = torch.cat((batch_pos_triples, batch_neg_triples), dim=0)  # (current_batch_size + current_batch_size * neg_num, 3)
                targets = torch.cat((pos_targets, neg_targets), dim=0)  # (current_batch_size + current_batch_size * neg_num)
                loss = rgcn_lp(edge_index=self.graph.edge_index, edge_type=self.graph.edge_type, triple_batch=batch_triples)  # (current_batch_size + current_batch_size * neg_num)
                batch_loss = criterion(input=loss, target=targets)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            print("- epoch {}, loss {}, time {}".format(epoch, epoch_loss, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            if lowest_loss > epoch_loss:
                lowest_loss = epoch_loss
                torch.save(rgcn_lp.state_dict(), self.model_path)
                print("model saved to {}".format(self.model_path))

    '''
    def ranking(self, name):
        triples = self.triples[name]  # validation/testing triples, (num_valid/test_triples, 3)
        # predict heads (num_valid/test_triples, num_entities, 3)
        triples[:, 1:3].unsqueeze(1)   # (num_valid/test_triples, 1, 2)
        torch.arange(self.count["entity"]).view(1, self.count["entity"], 1)  # (1, num_entities, 1)
        torch.cat(())
        # predict tails
    '''


# do link prediction
if __name__ == "__main__":
    rgcn_main = RgcnMain()
    rgcn_main.main()

