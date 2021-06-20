## Graph Convolutional Networks-based Link Prediction

Two popular Graph Convolutional Networks are implemented:

- Relational Graph Convolutional Network (RGCN), from [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103),

- Composition-based Graph Convolutional Network (CompGCN), from [Composition-based Multi-Relational Graph Convolutional Networks](https://openreview.net/pdf?id=BylA_C4tPr).

They have been evaluated in link prediction experiments. More details are to be given.

They are implemented based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric). With self-explanatory comments, the code is beginner-friendly. But if you are not familiar with Pytorch Geometric, checking out these [Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) could be a good start. 

----
### Environment

| python | torch | torch-scatter | torch-geometric |
| ---- | ---- | ---- | ---- |
| 3.7.10 | 1.8.0 | 2.0.6 | 1.6.3 |

Please refer to [Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the installation of torch-scatter and torch-geometric. 

----
### Dataset - FB15K237

| file | first line | other lines |
| ---- | ---- | ---- |
| entity2id.txt | number of entities: 14,541 | entity_name + '\t' + entity_id |
| relation2id.txt | number of relations: 237 | relation_name + '\t' + relation_id |
| train2id.txt, valid2id.txt, test2id.txt | number of training/ validation/ testing triples: 272,115/ 17,535/ 20,466 | head_entity_id + ' ' + tail_entity_id + ' ' + relation_id |

----

<img src="https://www.seekpng.com/png/detail/66-668670_board-under-construction-sign.png" alt="Board Under Construction Sign@seekpng.com">

----
### Contact

https://www.ifi.uzh.ch/en/ddis/people/ruijie.html
