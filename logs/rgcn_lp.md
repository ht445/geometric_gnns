-----
### Running Time: `2021-06-21 19:23:45`
#### configuration
- load data from `../data/FB15K237/`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `24`
- rgcn aggregation scheme: `add`
- train batch size: `20480`
- validation/test batch size: `500`
- learning rate: `0.01`
- number of epochs: `100`
- number of negative triples: `5`
- number of entities: `14541`
- number of relations: `238`
- number of training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
- model parameters: `['entity_embeds', 'rgcn.bases', 'rgcn.coefficients']`
#### training
- epoch `0`, loss `8.840275764465332`, time `2021-06-21 19:24:50`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `4952.5224609375`  |  `4210.560546875`  |  `4581.54150390625`  |  
|  mean reciprocal rank (MRR)  |  `0.003678957698866725`  |  `0.013247016817331314`  |  `0.008462986908853054`  |  
|  hit@1  |  `0.0013333334354683757`  |  `0.005555554758757353`  |  `0.003444444155320525`  |  
|  hit@3  |  `0.0029999995604157448`  |  `0.009388888254761696`  |  `0.00619444390758872`  |  
|  hit@10  |  `0.005611110478639603`  |  `0.02257142774760723`  |  `0.014091269113123417`  |  
- epoch `1`, loss `4.752285957336426`, time `2021-06-21 19:45:40`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `2`, loss `2.836796998977661`, time `2021-06-21 19:46:44`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `3`, loss `1.9770958423614502`, time `2021-06-21 19:47:49`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `4`, loss `1.6688930988311768`, time `2021-06-21 19:48:53`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `5`, loss `1.4897403717041016`, time `2021-06-21 19:49:57`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `628.0924682617188`  |  `253.35888671875`  |  `440.7256774902344`  |  
|  mean reciprocal rank (MRR)  |  `0.05843359977006912`  |  `0.2041432112455368`  |  `0.13128840923309326`  |  
|  hit@1  |  `0.026406098157167435`  |  `0.14815160632133484`  |  `0.08727885037660599`  |  
|  hit@3  |  `0.049867644906044006`  |  `0.20111851394176483`  |  `0.12549307942390442`  |  
|  hit@10  |  `0.11573640257120132`  |  `0.31062525510787964`  |  `0.21318082511425018`  |  
- epoch `6`, loss `1.3806662559509277`, time `2021-06-21 20:10:55`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `7`, loss `1.2872470617294312`, time `2021-06-21 20:11:59`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `8`, loss `1.2309467792510986`, time `2021-06-21 20:13:03`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `9`, loss `1.177661657333374`, time `2021-06-21 20:14:08`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `10`, loss `1.1371405124664307`, time `2021-06-21 20:15:12`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `581.64794921875`  |  `223.9052734375`  |  `402.776611328125`  |  
|  mean reciprocal rank (MRR)  |  `0.07140128314495087`  |  `0.21718698740005493`  |  `0.1442941427230835`  |  
|  hit@1  |  `0.03494443744421005`  |  `0.15495522320270538`  |  `0.09494982659816742`  |  
|  hit@3  |  `0.061611108481884`  |  `0.21544703841209412`  |  `0.13852907717227936`  |  
|  hit@10  |  `0.14043651521205902`  |  `0.34711143374443054`  |  `0.24377396702766418`  |  
- epoch `11`, loss `1.106589436531067`, time `2021-06-21 20:35:53`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `12`, loss `1.0720961093902588`, time `2021-06-21 20:36:57`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `13`, loss `1.0576218366622925`, time `2021-06-21 20:38:02`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `14`, loss `1.0273511409759521`, time `2021-06-21 20:39:07`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `15`, loss `1.008328914642334`, time `2021-06-21 20:40:12`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `589.605224609375`  |  `223.8890838623047`  |  `406.7471618652344`  |  
|  mean reciprocal rank (MRR)  |  `0.07943528890609741`  |  `0.22660671174526215`  |  `0.15302100777626038`  |  
|  hit@1  |  `0.0419091060757637`  |  `0.16385938227176666`  |  `0.10288424789905548`  |  
|  hit@3  |  `0.07366686314344406`  |  `0.2249407172203064`  |  `0.14930379390716553`  |  
|  hit@10  |  `0.14558011293411255`  |  `0.3567639887332916`  |  `0.2511720657348633`  |  
- epoch `16`, loss `0.9889218211174011`, time `2021-06-21 21:00:51`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `17`, loss `0.9744088649749756`, time `2021-06-21 21:01:55`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `18`, loss `0.9595491290092468`, time `2021-06-21 21:02:59`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `19`, loss `0.9475245475769043`, time `2021-06-21 21:04:03`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `20`, loss `0.9392406940460205`, time `2021-06-21 21:05:08`  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
-----
### Running Time: `2021-06-21 21:16:09`
#### configuration
- load data from `../data/FB15K237/`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `64`
- rgcn aggregation scheme: `add`
- train batch size: `20480`
- validation/test batch size: `500`
- learning rate: `0.01`
- number of epochs: `100`
- number of negative triples: `32`
- number of entities: `14541`
- number of relations: `238`
- number of training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
- model parameters: `['entity_embeds', 'rgcn.bases', 'rgcn.coefficients']`
#### training
- epoch `0`, loss `4.953436374664307`, time `2021-06-21 21:19:31`  
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `4805.84765625`  |  `3269.561767578125`  |  `4037.70458984375`  |  
|  mean reciprocal rank (MRR)  |  `0.0037554029840976`  |  `0.01962614804506302`  |  `0.011690775863826275`  |  
|  hit@1  |  `0.0016666668234393`  |  `0.01016722247004509`  |  `0.005916944704949856`  |  
|  hit@3  |  `0.0025555556640028954`  |  `0.015611778013408184`  |  `0.00908366683870554`  |  
|  hit@10  |  `0.004777777474373579`  |  `0.03168376162648201`  |  `0.018230769783258438`  |  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `1`, loss `1.7386029958724976`, time `2021-06-21 21:43:04`  
- epoch `2`, loss `1.2340800762176514`, time `2021-06-21 21:46:09`  
- epoch `3`, loss `0.9344106912612915`, time `2021-06-21 21:49:14`  
- epoch `4`, loss `0.790216863155365`, time `2021-06-21 21:52:19`  
- epoch `5`, loss `0.6979449391365051`, time `2021-06-21 21:55:27`  
validation results  
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `667.6615600585938`  |  `271.1745300292969`  |  `469.41802978515625`  |  
|  mean reciprocal rank (MRR)  |  `0.06230838969349861`  |  `0.21457646787166595`  |  `0.13844242691993713`  |  
|  hit@1  |  `0.031184431165456772`  |  `0.15955661237239838`  |  `0.09537052363157272`  |  
|  hit@3  |  `0.05547909066081047`  |  `0.21259380877017975`  |  `0.13403645157814026`  |  
|  hit@10  |  `0.11410829424858093`  |  `0.32178154587745667`  |  `0.2179449200630188`  |  
model saved to `../pretrained/FB15K237/rgcn_lp.pt`   
- epoch `6`, loss `0.6359087824821472`, time `2021-06-21 22:19:14`  
