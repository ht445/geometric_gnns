# Graph Convolutional Networks-based Link Prediction

## Overview

Two popular Graph Convolutional Networks (GCNs) are implemented:

- [Relational Graph Convolutional Networks (RGCNs)](https://arxiv.org/abs/1703.06103)

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/rgcn.png" alt="RGCN Arch" width="40%"> </p>
> Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

- [Composition-based Graph Convolutional Networks (CompGCNs)](https://openreview.net/pdf?id=BylA_C4tPr).

> <p align="center"> <img src="https://github.com/ruijie-wang-uzh/geometric_gnns/blob/master/others/compgcn.png" alt="CompGCN Arch" width="50%"> </p>
> Chiang, Wei-Lin, et al. "Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

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

| Model | MRR | MR | Hit@1 | Hit@3 | Hit@10 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [RGCN]((https://arxiv.org/abs/1703.06103)) | 0.248 | - | 0.153 | 0.258 | 0.414 | 
| [CompGCN](https://openreview.net/pdf?id=BylA_C4tPr) | 0.355 | 197 | 0.264 | 0.390 | 0.535 |

## Experimental Results

### RGCN ([commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/5015ff55862741ed2c8f1b271b24645771ed951e))

- Testing results of the rgcn implementation:

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean reciprocal rank (MRR)  |  `0.2861877679824829`  |  `0.30698487162590027`  |  **`0.2965863347053528`**  |  
|  mean rank (MR)  |  `1139.7838134765625`  |  `901.492919921875`  |  `1020.6383666992188`  |  
|  hit@1  |  `0.2779405117034912`  |  `0.2906406819820404`  |  **`0.284290611743927`**  |  
|  hit@3  |  `0.2817016839981079`  |  `0.3014359176158905`  |  **`0.2915688157081604`**  |  
|  hit@10  |  `0.29752829670906067`  |  `0.33006083965301514`  |  `0.3137945532798767`  |

- The results of MRR, hit@1, and hit@3 are better than those reported in the original paper.
- The model was tested after 17 epochs. I believe the performance can be improved with more epochs.
- The training takes about 40 minutes on a GeForce RTX 2080 Ti GPU with 11019 MiB Memory.
- [Commit of this training setting](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/5015ff55862741ed2c8f1b271b24645771ed951e)
- [Code](https://github.com/ruijie-wang-uzh/geometric_gnns/tree/bb8e468845d5a24fc4b20cf70fb4c8902d3a2fa1/rgcn)
- [Running log]()
- [Pretrained Model]()

- CompGCN ([commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/407e6699a42ee5b7c57cb0251eb69a8e25fe7079))

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `673.3828735351562`  |  `2129.835693359375`  |  `1401.6092529296875`  |  
|  mean reciprocal rank (MRR)  |  `0.12702743709087372`  |  `0.10114094614982605`  |  `0.11408419162034988`  |  
|  hit@1  |  `0.0467463880777359`  |  `0.05306652560830116`  |  `0.04990645498037338`  |  
|  hit@3  |  `0.18180181086063385`  |  `0.11458759009838104`  |  `0.14819470047950745`  |  
|  hit@10  |  `0.23247618973255157`  |  `0.1872071623802185`  |  `0.20984166860580444`  |