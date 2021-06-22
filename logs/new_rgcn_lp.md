-----
### Running Time: `2021-06-22 23:32:52`
#### configuration
- load data from `../data/FB15K237/`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `100`
- rgcn aggregation scheme: `add`
- train batch size: `20480`
- validation/test batch size: `500`
- learning rate: `0.01`
- number of epochs: `100`
- number of negative triples: `16`
- number of entities: `14541`
- number of relations: `475`
- number of training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
- patience: `2`
- model parameters: `['entity_embeds', 'rgcn.bases', 'rgcn.coefficients']`
#### training
- epoch `0`, loss `5.63094425201416`, time `2021-06-22 23:35:53`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1682.8466796875`  |  `1407.0023193359375`  |  `1544.924560546875`  |  
|  mean reciprocal rank (MRR)  |  `0.02472447045147419`  |  `0.09561304748058319`  |  `0.060168758034706116`  |  
|  hit@1  |  `0.010778222233057022`  |  `0.05910318344831467`  |  `0.034940704703330994`  |  
|  hit@3  |  `0.020889557898044586`  |  `0.09634128212928772`  |  `0.05861542001366615`  |  
|  hit@10  |  `0.04405722767114639`  |  `0.16581743955612183`  |  `0.10493732988834381`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `1.7339372634887695`, time `2021-06-23 00:07:14`  
