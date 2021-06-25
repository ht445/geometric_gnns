## Graph Convolutional Networks-based Link Prediction

Two popular Graph Convolutional Networks (GCNs) are implemented:

- Relational Graph Convolutional Network (RGCN), from [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103),

- Composition-based Graph Convolutional Network (CompGCN), from [Composition-based Multi-Relational Graph Convolutional Networks](https://openreview.net/pdf?id=BylA_C4tPr).

They have been evaluated in link prediction experiments.

They are implemented based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric). With self-explanatory comments, the code is beginner-friendly. But if you are not familiar with Pytorch Geometric, checking out these [Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) could be a good start. 

This work is still in progress, so I would be very grateful if you can find any bug or give me any feedback. Cheers!

----
### Environment

| python | torch | torch-scatter | torch-geometric |
| ---- | ---- | ---- | ---- |
| `3.7.10` | `1.8.0` | `2.0.6` | `1.6.3` |

Please refer to [Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the installation of torch-scatter and torch-geometric. 

----
### Dataset - FB15K237

| file | first line | other lines |
| ---- | ---- | ---- |
| `entity2id.txt` | number of entities: `14,541` | `entity_name + '\t' + entity_id` |
| `relation2id.txt` | number of relations: `237` | `relation_name + '\t' + relation_id` |
| `train2id.txt`, `valid2id.txt`, `test2id.txt` | number of training, validation, and testing triples: `272,115`, `17,535`, `20,466` | `head_entity_id + ' ' + tail_entity_id + ' ' + relation_id` |

----
### Experimental Results

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

----
