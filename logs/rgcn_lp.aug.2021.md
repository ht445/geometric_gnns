-----
### Running - `2021-08-24 00:54:07`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `100`
- number of negative triples: `1`
- learning rate: `0.01`
- number of bases: `64`
- rgcn aggregation scheme: `add`
- number of subgraphs: 20
- training subgraph batch size: `2`
- early-stop patience: `5`
- number of epochs: `100`
- validation frequency: `2`
- validation/test triple batch size: `500`
#### Preparing Data
-----
### Running - `2021-08-24 01:17:23`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `100`
- number of negative triples: `1`
- learning rate: `0.01`
- number of bases: `64`
- rgcn aggregation scheme: `add`
- number of subgraphs: 20
- training subgraph batch size: `2`
- early-stop patience: `5`
- number of epochs: `100`
- validation frequency: `2`
- validation/test triple batch size: `100`
- device: `cuda:3`
#### Preparing Data
- number of entities: `14541`
- number of original relations: `237`
- number of original training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
Computing METIS partitioning...
Done!
#### Model Training and Validation
- model parameters: `['ent_embeds', 'rgcn1.coefficients', 'rgcn1.bases', 'rgcn2.coefficients', 'rgcn2.bases', 'distmult.rel_embeds']`
- epoch `0`, loss `8.021968841552734`, time `2021-08-24 01:18:31`  
-----
### Running - `2021-08-24 01:21:16`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `100`
- number of negative triples: `1`
- learning rate: `0.01`
- number of bases: `64`
- rgcn aggregation scheme: `add`
- number of subgraphs: 20
- training subgraph batch size: `2`
- early-stop patience: `5`
- number of epochs: `100`
- validation frequency: `2`
- validation/test triple batch size: `100`
- device: `cuda:3`
#### Preparing Data
- number of entities: `14541`
- number of original relations: `237`
- number of original training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
Computing METIS partitioning...
Done!
#### Model Training and Validation
- model parameters: `['ent_embeds', 'rgcn1.coefficients', 'rgcn1.bases', 'rgcn2.coefficients', 'rgcn2.bases', 'distmult.rel_embeds']`
- epoch `0`, loss `6.977053642272949`, time `2021-08-24 01:22:21`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `6093.185546875`  |  `6283.02001953125`  |  `6188.1025390625`  |  
|  mean reciprocal rank (MRR)  |  `0.004644589498639107`  |  `0.00404266407713294`  |  `0.004343627020716667`  |  
|  hit@1  |  `0.0036937540862709284`  |  `0.0034664818085730076`  |  `0.00358011806383729`  |  
|  hit@3  |  `0.003977844957262278`  |  `0.0034664818085730076`  |  `0.0037221633829176426`  |  
|  hit@10  |  `0.004659662488847971`  |  `0.003750572446733713`  |  `0.004205117467790842`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `14.975062370300293`, time `2021-08-24 02:06:46`  
- epoch `2`, loss `15.602034568786621`, time `2021-08-24 02:06:53`  
- validation results  at epoch `2`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `6332.609375`  |  `6388.9609375`  |  `6360.78515625`  |  
|  mean reciprocal rank (MRR)  |  `0.004541188944131136`  |  `0.00475578336045146`  |  `0.004648486152291298`  |  
|  hit@1  |  `0.003465907881036401`  |  `0.003636362263932824`  |  `0.0035511350724846125`  |  
|  hit@3  |  `0.0038636347744613886`  |  `0.003921026363968849`  |  `0.003892330452799797`  |  
|  hit@10  |  `0.00528408819809556`  |  `0.006080116145312786`  |  `0.005682102404534817`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `2`   
- epoch `3`, loss `14.451539039611816`, time `2021-08-24 02:50:38`  
- epoch `4`, loss `18.756351470947266`, time `2021-08-24 02:50:45`  
- validation results  at epoch `4`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5899.16357421875`  |  `6173.6611328125`  |  `6036.412109375`  |  
|  mean reciprocal rank (MRR)  |  `0.010710290633141994`  |  `0.01102652307599783`  |  `0.010868406854569912`  |  
|  hit@1  |  `0.009491500444710255`  |  `0.009834131225943565`  |  `0.009662816300988197`  |  
|  hit@3  |  `0.009661954827606678`  |  `0.01017561461776495`  |  `0.009918784722685814`  |  
|  hit@10  |  `0.011253435164690018`  |  `0.010800614021718502`  |  `0.011027025058865547`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `4`   
- epoch `5`, loss `10.418166160583496`, time `2021-08-24 03:34:26`  
- epoch `6`, loss `14.052746772766113`, time `2021-08-24 03:34:32`  
- validation results  at epoch `6`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5370.9677734375`  |  `5402.61572265625`  |  `5386.7919921875`  |  
|  mean reciprocal rank (MRR)  |  `0.011739080771803856`  |  `0.012514978647232056`  |  `0.012127029709517956`  |  
|  hit@1  |  `0.010114203207194805`  |  `0.00988693069666624`  |  `0.010000566951930523`  |  
|  hit@3  |  `0.011193748563528061`  |  `0.012614201754331589`  |  `0.011903975158929825`  |  
|  hit@10  |  `0.012443747371435165`  |  `0.01460340991616249`  |  `0.013523578643798828`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `6`   
- epoch `7`, loss `17.533798217773438`, time `2021-08-24 04:18:09`  
- epoch `8`, loss `24.729534149169922`, time `2021-08-24 04:18:15`  
- validation results  at epoch `8`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3388.79296875`  |  `3348.705322265625`  |  `3368.7490234375`  |  
|  mean reciprocal rank (MRR)  |  `0.055223990231752396`  |  `0.05626709386706352`  |  `0.05574554204940796`  |  
|  hit@1  |  `0.0539490208029747`  |  `0.05406207963824272`  |  `0.05400554835796356`  |  
|  hit@3  |  `0.05417743697762489`  |  `0.05525697395205498`  |  `0.054717205464839935`  |  
|  hit@10  |  `0.05537060648202896`  |  `0.05775925889611244`  |  `0.05656493455171585`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `8`   
- epoch `9`, loss `18.8111572265625`, time `2021-08-24 05:01:33`  
- epoch `10`, loss `26.87137222290039`, time `2021-08-24 05:01:39`  
- validation results  at epoch `10`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3471.7763671875`  |  `3362.998291015625`  |  `3417.38720703125`  |  
|  mean reciprocal rank (MRR)  |  `0.06367500126361847`  |  `0.0639328882098198`  |  `0.06380394101142883`  |  
|  hit@1  |  `0.06282709538936615`  |  `0.06282709538936615`  |  `0.06282709538936615`  |  
|  hit@3  |  `0.06282709538936615`  |  `0.06288391351699829`  |  `0.06285550445318222`  |  
|  hit@10  |  `0.06288391351699829`  |  `0.06299754977226257`  |  `0.06294073164463043`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `10`   
- epoch `11`, loss `26.518352508544922`, time `2021-08-24 05:44:59`  
- epoch `12`, loss `29.913806915283203`, time `2021-08-24 05:45:09`  
- validation results  at epoch `12`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `4128.09716796875`  |  `3990.892822265625`  |  `4059.4951171875`  |  
|  mean reciprocal rank (MRR)  |  `0.12136346846818924`  |  `0.12170854210853577`  |  `0.1215360015630722`  |  
|  hit@1  |  `0.12084893882274628`  |  `0.12090575695037842`  |  `0.12087734788656235`  |  
|  hit@3  |  `0.12084893882274628`  |  `0.12096257507801056`  |  `0.12090575695037842`  |  
|  hit@10  |  `0.12090575695037842`  |  `0.12136029452085495`  |  `0.12113302946090698`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `12`   
- epoch `13`, loss `37.36430358886719`, time `2021-08-24 06:28:24`  
- epoch `14`, loss `28.647132873535156`, time `2021-08-24 06:28:30`  
- validation results  at epoch `14`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3475.376708984375`  |  `3982.289794921875`  |  `3728.833251953125`  |  
|  mean reciprocal rank (MRR)  |  `0.133106529712677`  |  `0.13299477100372314`  |  `0.13305065035820007`  |  
|  hit@1  |  `0.13205595314502716`  |  `0.13205595314502716`  |  `0.13205595314502716`  |  
|  hit@3  |  `0.13205595314502716`  |  `0.13205595314502716`  |  `0.13205595314502716`  |  
|  hit@10  |  `0.13245368003845215`  |  `0.13217075169086456`  |  `0.13231220841407776`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `14`   
- epoch `15`, loss `40.70518493652344`, time `2021-08-24 07:11:45`  
- epoch `16`, loss `30.98896598815918`, time `2021-08-24 07:11:51`  
- validation results  at epoch `16`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3535.820556640625`  |  `3427.875732421875`  |  `3481.84814453125`  |  
|  mean reciprocal rank (MRR)  |  `0.16173449158668518`  |  `0.16143189370632172`  |  `0.16158318519592285`  |  
|  hit@1  |  `0.16067597270011902`  |  `0.1605055183172226`  |  `0.1605907380580902`  |  
|  hit@3  |  `0.16101689636707306`  |  `0.16056233644485474`  |  `0.1607896089553833`  |  
|  hit@10  |  `0.1617555469274521`  |  `0.16096064448356628`  |  `0.1613580882549286`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `16`   
- epoch `17`, loss `33.740482330322266`, time `2021-08-24 07:55:08`  
- epoch `18`, loss `39.900611877441406`, time `2021-08-24 07:55:14`  
- validation results  at epoch `18`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3711.210693359375`  |  `3635.89453125`  |  `3673.552734375`  |  
|  mean reciprocal rank (MRR)  |  `0.12873639166355133`  |  `0.1288704127073288`  |  `0.12880340218544006`  |  
|  hit@1  |  `0.1281564086675644`  |  `0.1281564086675644`  |  `0.1281564086675644`  |  
|  hit@3  |  `0.1281564086675644`  |  `0.1281564086675644`  |  `0.1281564086675644`  |  
|  hit@10  |  `0.12827003002166748`  |  `0.12832742929458618`  |  `0.12829872965812683`  |  
   
- epoch `19`, loss `34.872493743896484`, time `2021-08-24 08:38:28`  
- epoch `20`, loss `33.08846664428711`, time `2021-08-24 08:38:34`  
