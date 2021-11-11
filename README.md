# Graph Convolutional Networks-based Link Prediction

## Overview

Two popular Graph Convolutional Networks (GCNs) are implemented:

- [Composition-based Graph Convolutional Networks (CompGCNs)](https://arxiv.org/abs/1911.03082).

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/compgcn.png" alt="CompGCN Arch" width="50%"> </p>
> Vashishth, Shikhar, et al. "Composition-based Multi-Relational Graph Convolutional Networks." International Conference on Learning Representations. 2019.

- [Relational Graph Convolutional Networks (RGCNs)](https://arxiv.org/abs/1703.06103)

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/rgcn.png" alt="RGCN Arch" width="40%"> </p>
> Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

The implementation is based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric).

To train on GPUs with limited memory, the training graph can be partitioned into subgraphs based on the method proposed in [ClusterGCN](https://dl.acm.org/doi/abs/10.1145/3292500.3330925).  

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/cluster_gcn.png" alt="Knowledge Graph Partitioning" width="60%"> </p>
> Chiang, Wei-Lin, et al. "Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

## Environment

| python | pytorch | pytorch-scatter | pytorch-geometric |
| ---- | ---- | ---- | ---- |
| `3.8.11` | `1.9.0` | `2.0.8` | `1.7.2` |

- Detailed requirements: [environment.yml](https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/environment.yml)

## Dataset

### FB15K237

| file | first line | other lines |
| ---- | ---- | ---- |
| `entity2id.txt` | number of entities: 14,541 | `entity_name + '\t' + entity_id` |
| `relation2id.txt` | number of relations: 237 | `relation_name + '\t' + relation_id` |
| `train2id.txt`, `valid2id.txt`, `test2id.txt` | number of training, validation, and testing triples: 272,115, 17,535, 20,466 | `head_entity_id + ' ' + tail_entity_id + ' ' + relation_id` |

### Performance reported in the papers

| Model | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [CompGCN](https://arxiv.org/abs/1911.03082) | 0.355 | 197 | 0.264 | 0.390 | 0.535 |
| [RGCN](https://arxiv.org/abs/1703.06103) | 0.248 | - | 0.153 | 0.258 | 0.414 | 

## Experimental Results

### CompGCN

|  metric  |  head  |  tail  |  mean  |
|  ----  |  ----  |  ----  |  ----  |
|  mean reciprocal rank (MRR)  |  `0.15498019754886627`  |  `0.3260224759578705`  |  `0.24050134420394897`  |
|  mean rank (MR)  |  `398.7995300292969`  |  `201.8601531982422`  |  `300.329833984375`  |
|  hits@1  |  `0.0870712401055409`  |  `0.23370468093423238`  |  `0.16038796051988663`  |
|  hits@3  |  `0.1679370663539529`  |  `0.35922994234339883`  |  `0.26358350434867583`  |
|  hits@10  |  `0.2876478061174631`  |  `0.5122153816085214`  |  `0.39993159386299226`  |

- The training (50 epochs) took about 14 hours on a GeForce RTX 2080 Ti GPU with 11019 MiB Memory.
- [Running log]()
- [Pretrained model]()

### RGCN

... ...