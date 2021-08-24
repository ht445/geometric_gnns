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

- FB15K237

| file | first line | other lines |
| ---- | ---- | ---- |
| `entity2id.txt` | number of entities: `14,541` | `entity_name + '\t' + entity_id` |
| `relation2id.txt` | number of relations: `237` | `relation_name + '\t' + relation_id` |
| `train2id.txt`, `valid2id.txt`, `test2id.txt` | number of training, validation, and testing triples: `272,115`, `17,535`, `20,466` | `head_entity_id + ' ' + tail_entity_id + ' ' + relation_id` |

- Performance reported in the papers

| Model | MRR | MR | Hit@1 | Hit@3 | Hit@10 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RGCN | 0.248 | - | 0.153 | 0.258 | 0.414 | 
| CompGCN | 0.355 | 197 | 0.535 | 0.390 | 0.264 |

## Experimental Results

- Note: the following only reports in-progress results; model architectures and hyperparameters are not well set. 

- RGCN ([corresponding commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/90bc1f39a6600498e3adc557dca9d51e16abbc15))

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `393.8387145996094`  |  `228.19479370117188`  |  `311.0167541503906`  |  
|  mean reciprocal rank (MRR)  |  `0.11209864914417267`  |  `0.2061285823583603`  |  `0.15911361575126648`  |  
|  hit@1  |  `0.049701616168022156`  |  `0.11677433550357819`  |  `0.08323797583580017`  |  
|  hit@3  |  `0.11515237390995026`  |  `0.2234063744544983`  |  `0.16927936673164368`  |  
|  hit@10  |  `0.24000363051891327`  |  `0.396608829498291`  |  `0.31830623745918274`  |  

- CompGCN ([corresponding commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/407e6699a42ee5b7c57cb0251eb69a8e25fe7079))

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `673.3828735351562`  |  `2129.835693359375`  |  `1401.6092529296875`  |  
|  mean reciprocal rank (MRR)  |  `0.12702743709087372`  |  `0.10114094614982605`  |  `0.11408419162034988`  |  
|  hit@1  |  `0.0467463880777359`  |  `0.05306652560830116`  |  `0.04990645498037338`  |  
|  hit@3  |  `0.18180181086063385`  |  `0.11458759009838104`  |  `0.14819470047950745`  |  
|  hit@10  |  `0.23247618973255157`  |  `0.1872071623802185`  |  `0.20984166860580444`  |  

- CompGCN + RGCN ([corresponding commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/67e91668e2c3f8a36a736bc924799a1e5cf5a8e8))
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `358.2768859863281`  |  `185.40789794921875`  |  `271.8424072265625`  |  
|  mean reciprocal rank (MRR)  |  `0.09944625943899155`  |  `0.23323273658752441`  |  `0.16633950173854828`  |  
|  hit@1  |  `0.04600272700190544`  |  `0.14427782595157623`  |  `0.09514027833938599`  |  
|  hit@3  |  `0.09846094995737076`  |  `0.25662872195243835`  |  `0.17754483222961426`  |  
|  hit@10  |  `0.20743079483509064`  |  `0.41294726729393005`  |  `0.31018903851509094`  |  
   
----
