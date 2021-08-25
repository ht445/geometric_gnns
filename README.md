# Graph Convolutional Networks-based Link Prediction

## Overview

Two popular Graph Convolutional Networks (GCNs) are implemented:

- [Relational Graph Convolutional Networks (RGCNs)](https://arxiv.org/abs/1703.06103)

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/rgcn.png" alt="RGCN Arch" width="40%"> </p>
> Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

- [Composition-based Graph Convolutional Networks (CompGCNs)](https://arxiv.org/abs/1911.03082).

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/compgcn.png" alt="CompGCN Arch" width="50%"> </p>
> Vashishth, Shikhar, et al. "Composition-based Multi-Relational Graph Convolutional Networks." International Conference on Learning Representations. 2019.

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
| [RGCN](https://arxiv.org/abs/1703.06103) | 0.248 | - | 0.153 | 0.258 | 0.414 | 
| [CompGCN](https://arxiv.org/abs/1911.03082) | 0.355 | 197 | 0.264 | 0.390 | 0.535 |

## Experimental Results

### RGCN

- Testing results of the rgcn implementation:

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean reciprocal rank (MRR)  |  `0.26442909240722656`  |  `0.28572818636894226`  |  **`0.2750786542892456`**  |
|  hits@1  |  `0.2517834457148441`  |  `0.2642919964819701`  |  **`0.2580377210984071`**  |  
|  hits@3  |  `0.2595524284178638`  |  `0.2789015928857618`  |  **`0.26922701065181276`**  |  
|  hits@10  |  `0.2842274992670771`  |  `0.32199745920062545`  |  `0.30311247923385126`  |  

- The results of MRR, hits@1, and hits@3 are better than those reported in the original paper.
- The training takes about 1 hour on a GeForce RTX 2080 Ti GPU with 11019 MiB Memory.
- [Running log](https://github.com/ruijie-wang-uzh/geometric_gnns/blob/2dc89c75fe480c5379408aea972d38217dc62e5d/logs/rgcn_lp.aug.2021.md)
- [Pretrained model](https://github.com/ruijie-wang-uzh/geometric_gnns/blob/946622e922547515373291267745c8594ce1cf6d/pretrained/FB15K237/rgcn_lp.pt)

### CompGCN

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `673.3828735351562`  |  `2129.835693359375`  |  `1401.6092529296875`  |  
|  mean reciprocal rank (MRR)  |  `0.12702743709087372`  |  `0.10114094614982605`  |  `0.11408419162034988`  |  
|  hits@1  |  `0.0467463880777359`  |  `0.05306652560830116`  |  `0.04990645498037338`  |  
|  hits@3  |  `0.18180181086063385`  |  `0.11458759009838104`  |  `0.14819470047950745`  |  
|  hits@10  |  `0.23247618973255157`  |  `0.1872071623802185`  |  `0.20984166860580444`  |

- [commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/407e6699a42ee5b7c57cb0251eb69a8e25fe7079)