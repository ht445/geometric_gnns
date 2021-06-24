## Graph Convolutional Networks-based Link Prediction

Two popular Graph Convolutional Networks (GCN) are implemented:

- Relational Graph Convolutional Network (RGCN), from [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103),

- Composition-based Graph Convolutional Network (CompGCN), from [Composition-based Multi-Relational Graph Convolutional Networks](https://openreview.net/pdf?id=BylA_C4tPr).

They have been evaluated in link prediction experiments. More details are to be given.

They are implemented based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric). With self-explanatory comments, the code is beginner-friendly. But if you are not familiar with Pytorch Geometric, checking out these [Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) could be a good start. 

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

- RGCN ([corresponding commit](https://github.com/ruijie-wang-uzh/geometric_gnns/commit/90bc1f39a6600498e3adc557dca9d51e16abbc15))

|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `393.8387145996094`  |  `228.19479370117188`  |  `311.0167541503906`  |  
|  mean reciprocal rank (MRR)  |  `0.11209864914417267`  |  `0.2061285823583603`  |  `0.15911361575126648`  |  
|  hit@1  |  `0.049701616168022156`  |  `0.11677433550357819`  |  `0.08323797583580017`  |  
|  hit@3  |  `0.11515237390995026`  |  `0.2234063744544983`  |  `0.16927936673164368`  |  
|  hit@10  |  `0.24000363051891327`  |  `0.396608829498291`  |  `0.31830623745918274`  |  


- Note: the link prediction experiment is just to showcase the application of GCNs. Neither the model architecture nor the parameter setting was fine-tuned. Therefore, the above results may look inferior. 

----

<img src="https://www.seekpng.com/png/detail/66-668670_board-under-construction-sign.png" alt="Board Under Construction Sign@seekpng.com">

----
### Contact

https://www.ifi.uzh.ch/en/ddis/people/ruijie.html
