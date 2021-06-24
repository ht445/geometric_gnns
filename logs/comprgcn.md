-----
### Running Time: `2021-06-24 22:09:07`
#### configuration
- load data from `../data/FB15K237/`
- operation: `RGCN`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `1024`
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
- norm: `2`
- CompRGCN instantiated
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_r_weight', 'compgcn.relation_l_weight']`
#### training
-----
### Running Time: `2021-06-24 22:33:48`
#### configuration
- load data from `../data/FB15K237/`
- operation: `RGCN`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
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
- norm: `2`
- CompRGCN instantiated
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_r_weight', 'compgcn.relation_l_weight']`
#### training
- epoch `0`, loss `13.733050346374512`, time `2021-06-24 22:38:26`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5328.37939453125`  |  `4930.23193359375`  |  `5129.3056640625`  |  
|  mean reciprocal rank (MRR)  |  `0.0033835810609161854`  |  `0.0026256958954036236`  |  `0.0030046384781599045`  |  
|  hit@1  |  `0.0005000000237487257`  |  `0.00022222223924472928`  |  `0.00036111113149672747`  |  
|  hit@3  |  `0.0018888890044763684`  |  `0.0011666667414829135`  |  `0.001527777872979641`  |  
|  hit@10  |  `0.006666665896773338`  |  `0.004555555060505867`  |  `0.005611110478639603`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `3.257871627807617`, time `2021-06-24 23:04:05`  
- epoch `2`, loss `2.128474473953247`, time `2021-06-24 23:08:45`  
- epoch `3`, loss `1.8006726503372192`, time `2021-06-24 23:13:20`  
- validation results  at epoch `3`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `877.2769775390625`  |  `569.6705932617188`  |  `723.4737548828125`  |  
|  mean reciprocal rank (MRR)  |  `0.04095003008842468`  |  `0.14807157218456268`  |  `0.09451080113649368`  |  
|  hit@1  |  `0.015460318885743618`  |  `0.09236232936382294`  |  `0.053911324590444565`  |  
|  hit@3  |  `0.03464285656809807`  |  `0.15424486994743347`  |  `0.09444386512041092`  |  
|  hit@10  |  `0.08454762399196625`  |  `0.2614189386367798`  |  `0.1729832887649536`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `3`   
- epoch `4`, loss `1.39711594581604`, time `2021-06-24 23:40:02`  
- epoch `5`, loss `1.2023670673370361`, time `2021-06-24 23:45:09`  
- epoch `6`, loss `1.1333123445510864`, time `2021-06-24 23:50:13`  
- validation results  at epoch `6`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `538.0274047851562`  |  `298.1578369140625`  |  `418.0926208496094`  |  
|  mean reciprocal rank (MRR)  |  `0.0649515762925148`  |  `0.20828750729560852`  |  `0.13661953806877136`  |  
|  hit@1  |  `0.026849202811717987`  |  `0.13175052404403687`  |  `0.07929986715316772`  |  
|  hit@3  |  `0.05914284661412239`  |  `0.2238687425851822`  |  `0.14150579273700714`  |  
|  hit@10  |  `0.13506348431110382`  |  `0.3633047342300415`  |  `0.24918410181999207`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `6`   
