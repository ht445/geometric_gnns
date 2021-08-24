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
- validation results  at epoch `20`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `3393.631103515625`  |  `3470.515869140625`  |  `3432.073486328125`  |  
|  mean reciprocal rank (MRR)  |  `0.16356422007083893`  |  `0.16369718313217163`  |  `0.16363069415092468`  |  
|  hit@1  |  `0.16293323040008545`  |  `0.16293323040008545`  |  `0.16293323040008545`  |  
|  hit@3  |  `0.16293323040008545`  |  `0.16293323040008545`  |  `0.16293323040008545`  |  
|  hit@10  |  `0.1629900485277176`  |  `0.1629900485277176`  |  `0.1629900485277176`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `20`   
- epoch `21`, loss `24.505348205566406`, time `2021-08-24 09:22:02`  
- epoch `22`, loss `23.339651107788086`, time `2021-08-24 09:22:11`  
-----
### Running - `2021-08-24 09:57:50`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `100`
- number of negative triples: `1`
- learning rate: `0.01`
- number of bases: `64`
- rgcn aggregation scheme: `add`
- number of subgraphs: 1
- training subgraph batch size: `1`
- early-stop patience: `5`
- number of epochs: `100`
- validation frequency: `5`
- validation/test triple batch size: `200`
- device: `cpu`
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
step `0`, time `2021-08-24 09:58:46`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `0`, loss `0.6931474804878235`, time `2021-08-24 10:00:28`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1717.9613037109375`  |  `1583.522705078125`  |  `1650.741943359375`  |  
|  mean reciprocal rank (MRR)  |  `0.01660340093076229`  |  `0.039434127509593964`  |  `0.028018765151500702`  |  
|  hit@1  |  `0.006504627875983715`  |  `0.021080292761325836`  |  `0.013792460784316063`  |  
|  hit@3  |  `0.012895617634057999`  |  `0.037100162357091904`  |  `0.02499788999557495`  |  
|  hit@10  |  `0.030450332909822464`  |  `0.07136286050081253`  |  `0.050906598567962646`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `0`   
step `0`, time `2021-08-24 10:47:52`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `1`, loss `0.693042516708374`, time `2021-08-24 10:48:53`  
step `0`, time `2021-08-24 10:48:53`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `2`, loss `3.9427173137664795`, time `2021-08-24 10:49:52`  
step `0`, time `2021-08-24 10:49:52`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `3`, loss `1.1019933223724365`, time `2021-08-24 10:50:51`  
step `0`, time `2021-08-24 10:50:51`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `4`, loss `0.6864475607872009`, time `2021-08-24 10:51:51`  
step `0`, time `2021-08-24 10:51:51`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `5`, loss `0.6778223514556885`, time `2021-08-24 10:52:49`  
- validation results  at epoch `5`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5514.228515625`  |  `5107.7021484375`  |  `5310.96533203125`  |  
|  mean reciprocal rank (MRR)  |  `0.0029825889505445957`  |  `0.017438778653740883`  |  `0.010210683569312096`  |  
|  hit@1  |  `0.0009090909152291715`  |  `0.008807956241071224`  |  `0.004858523607254028`  |  
|  hit@3  |  `0.0018749996088445187`  |  `0.017188437283039093`  |  `0.00953171867877245`  |  
|  hit@10  |  `0.005284089595079422`  |  `0.03264317661523819`  |  `0.018963633105158806`  |  
   
step `0`, time `2021-08-24 11:39:24`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `6`, loss `0.65230792760849`, time `2021-08-24 11:40:19`  
step `0`, time `2021-08-24 11:40:19`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `7`, loss `0.704973042011261`, time `2021-08-24 11:41:17`  
step `0`, time `2021-08-24 11:41:17`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `8`, loss `0.6272099614143372`, time `2021-08-24 11:42:14`  
step `0`, time `2021-08-24 11:42:14`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `9`, loss `0.6085300445556641`, time `2021-08-24 11:43:10`  
step `0`, time `2021-08-24 11:43:10`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `10`, loss `0.5529889464378357`, time `2021-08-24 11:44:08`  
- validation results  at epoch `10`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `2167.465576171875`  |  `2262.352783203125`  |  `2214.9091796875`  |  
|  mean reciprocal rank (MRR)  |  `0.014852961525321007`  |  `0.030923420563340187`  |  `0.022888191044330597`  |  
|  hit@1  |  `0.005595535971224308`  |  `0.014315678738057613`  |  `0.00995560735464096`  |  
|  hit@3  |  `0.01087962370365858`  |  `0.027754129841923714`  |  `0.01931687630712986`  |  
|  hit@10  |  `0.02471379190683365`  |  `0.056107450276613235`  |  `0.04041062295436859`  |  
   
step `0`, time `2021-08-24 12:31:09`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `11`, loss `0.549092173576355`, time `2021-08-24 12:32:07`  
step `0`, time `2021-08-24 12:32:07`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `12`, loss `0.49353599548339844`, time `2021-08-24 12:33:10`  
step `0`, time `2021-08-24 12:33:10`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `13`, loss `0.46418896317481995`, time `2021-08-24 12:34:07`  
step `0`, time `2021-08-24 12:34:07`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `14`, loss `0.41530027985572815`, time `2021-08-24 12:35:07`  
step `0`, time `2021-08-24 12:35:08`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `15`, loss `0.4176744520664215`, time `2021-08-24 12:36:08`  
- validation results  at epoch `15`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1282.3653564453125`  |  `962.9222412109375`  |  `1122.643798828125`  |  
|  mean reciprocal rank (MRR)  |  `0.022366320714354515`  |  `0.05532890558242798`  |  `0.03884761407971382`  |  
|  hit@1  |  `0.008122892118990421`  |  `0.02724219672381878`  |  `0.017682544887065887`  |  
|  hit@3  |  `0.01780933141708374`  |  `0.04533974826335907`  |  `0.031574539840221405`  |  
|  hit@10  |  `0.04161405190825462`  |  `0.10857315361499786`  |  `0.07509360462427139`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `15`   
step `0`, time `2021-08-24 13:52:32`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `16`, loss `0.3627837896347046`, time `2021-08-24 13:53:29`  
step `0`, time `2021-08-24 13:53:29`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `17`, loss `0.37551331520080566`, time `2021-08-24 13:54:31`  
step `0`, time `2021-08-24 13:54:31`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `18`, loss `0.35745835304260254`, time `2021-08-24 13:55:27`  
step `0`, time `2021-08-24 13:55:27`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `19`, loss `0.32545942068099976`, time `2021-08-24 13:56:25`  
step `0`, time `2021-08-24 13:56:25`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `20`, loss `0.30634555220603943`, time `2021-08-24 13:57:24`  
- validation results  at epoch `20`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `863.8429565429688`  |  `583.8685302734375`  |  `723.855712890625`  |  
|  mean reciprocal rank (MRR)  |  `0.030765511095523834`  |  `0.08669605851173401`  |  `0.05873078480362892`  |  
|  hit@1  |  `0.01178775355219841`  |  `0.05002785846590996`  |  `0.030907806009054184`  |  
|  hit@3  |  `0.021645788103342056`  |  `0.07761609554290771`  |  `0.049630939960479736`  |  
|  hit@10  |  `0.05624593794345856`  |  `0.15271300077438354`  |  `0.10447946935892105`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `20`   
step `0`, time `2021-08-24 14:43:17`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `21`, loss `0.28423404693603516`, time `2021-08-24 14:44:14`  
step `0`, time `2021-08-24 14:44:14`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `22`, loss `0.2695964276790619`, time `2021-08-24 14:45:07`  
step `0`, time `2021-08-24 14:45:08`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `23`, loss `0.26462456583976746`, time `2021-08-24 14:46:09`  
step `0`, time `2021-08-24 14:46:09`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `24`, loss `0.27470478415489197`, time `2021-08-24 14:47:11`  
step `0`, time `2021-08-24 14:47:11`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `25`, loss `0.24182917177677155`, time `2021-08-24 14:48:06`  
- validation results  at epoch `25`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `688.8226928710938`  |  `439.3372802734375`  |  `564.0799560546875`  |  
|  mean reciprocal rank (MRR)  |  `0.03831382095813751`  |  `0.10418438166379929`  |  `0.0712490975856781`  |  
|  hit@1  |  `0.012840796262025833`  |  `0.0546947605907917`  |  `0.03376777842640877`  |  
|  hit@3  |  `0.02787129394710064`  |  `0.0977075845003128`  |  `0.0627894401550293`  |  
|  hit@10  |  `0.08540232479572296`  |  `0.20290249586105347`  |  `0.14415240287780762`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `25`   
step `0`, time `2021-08-24 15:34:42`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `26`, loss `0.23825512826442719`, time `2021-08-24 15:35:38`  
step `0`, time `2021-08-24 15:35:39`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `27`, loss `0.2446528822183609`, time `2021-08-24 15:36:36`  
step `0`, time `2021-08-24 15:36:36`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `28`, loss `0.23794236779212952`, time `2021-08-24 15:37:32`  
step `0`, time `2021-08-24 15:37:32`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `29`, loss `0.2496793121099472`, time `2021-08-24 15:38:28`  
step `0`, time `2021-08-24 15:38:28`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `30`, loss `0.23305848240852356`, time `2021-08-24 15:39:24`  
- validation results  at epoch `30`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `622.9247436523438`  |  `384.70245361328125`  |  `503.8135986328125`  |  
|  mean reciprocal rank (MRR)  |  `0.06650286167860031`  |  `0.13742674887180328`  |  `0.1019648015499115`  |  
|  hit@1  |  `0.0385538749396801`  |  `0.08455029875040054`  |  `0.06155208498239517`  |  
|  hit@3  |  `0.05744589865207672`  |  `0.134928360581398`  |  `0.09618712961673737`  |  
|  hit@10  |  `0.12006271630525589`  |  `0.24044999480247498`  |  `0.18025635182857513`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `30`   
step `0`, time `2021-08-24 16:24:25`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `31`, loss `0.2255280315876007`, time `2021-08-24 16:25:19`  
step `0`, time `2021-08-24 16:25:19`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `32`, loss `0.21831725537776947`, time `2021-08-24 16:26:15`  
step `0`, time `2021-08-24 16:26:15`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `33`, loss `0.2055952548980713`, time `2021-08-24 16:27:12`  
step `0`, time `2021-08-24 16:27:13`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `34`, loss `0.19519543647766113`, time `2021-08-24 16:28:10`  
step `0`, time `2021-08-24 16:28:11`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `35`, loss `0.2008180022239685`, time `2021-08-24 16:29:12`  
- validation results  at epoch `35`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `570.1604614257812`  |  `320.3941345214844`  |  `445.27728271484375`  |  
|  mean reciprocal rank (MRR)  |  `0.05446374788880348`  |  `0.14604252576828003`  |  `0.1002531349658966`  |  
|  hit@1  |  `0.023552250117063522`  |  `0.0800258219242096`  |  `0.05178903788328171`  |  
|  hit@3  |  `0.04210252687335014`  |  `0.1512088030576706`  |  `0.09665566682815552`  |  
|  hit@10  |  `0.11781005561351776`  |  `0.28276991844177246`  |  `0.2002899944782257`  |  
   
step `0`, time `2021-08-24 17:14:43`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `36`, loss `0.18636471033096313`, time `2021-08-24 17:15:31`  
step `0`, time `2021-08-24 17:15:31`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `37`, loss `0.2028694897890091`, time `2021-08-24 17:16:27`  
step `0`, time `2021-08-24 17:16:27`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `38`, loss `0.1938260793685913`, time `2021-08-24 17:17:24`  
step `0`, time `2021-08-24 17:17:24`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `39`, loss `0.20172333717346191`, time `2021-08-24 17:18:20`  
step `0`, time `2021-08-24 17:18:21`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `40`, loss `0.20246386528015137`, time `2021-08-24 17:19:19`  
- validation results  at epoch `40`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `535.1431274414062`  |  `297.7659912109375`  |  `416.4545593261719`  |  
|  mean reciprocal rank (MRR)  |  `0.06250675022602081`  |  `0.17145751416683197`  |  `0.11698213219642639`  |  
|  hit@1  |  `0.02849351242184639`  |  `0.10773059725761414`  |  `0.06811205297708511`  |  
|  hit@3  |  `0.05287088453769684`  |  `0.17128992080688477`  |  `0.1120804026722908`  |  
|  hit@10  |  `0.1318836808204651`  |  `0.3043641149997711`  |  `0.2181238979101181`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `40`   
step `0`, time `2021-08-24 18:05:21`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `41`, loss `0.1816045194864273`, time `2021-08-24 18:06:17`  
step `0`, time `2021-08-24 18:06:17`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `42`, loss `0.18371926248073578`, time `2021-08-24 18:07:15`  
step `0`, time `2021-08-24 18:07:15`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `43`, loss `0.1766669601202011`, time `2021-08-24 18:08:16`  
step `0`, time `2021-08-24 18:08:16`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `44`, loss `0.16869711875915527`, time `2021-08-24 18:09:19`  
step `0`, time `2021-08-24 18:09:20`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `45`, loss `0.1638180911540985`, time `2021-08-24 18:10:15`  
- validation results  at epoch `45`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `508.9562683105469`  |  `280.6371765136719`  |  `394.7967224121094`  |  
|  mean reciprocal rank (MRR)  |  `0.06019549444317818`  |  `0.17959292232990265`  |  `0.11989420652389526`  |  
|  hit@1  |  `0.02621631696820259`  |  `0.1113123819231987`  |  `0.0687643513083458`  |  
|  hit@3  |  `0.05038932338356972`  |  `0.18425841629505157`  |  `0.1173238679766655`  |  
|  hit@10  |  `0.12449705600738525`  |  `0.3266202211380005`  |  `0.22555863857269287`  |  
   
- model saved to `../pretrained/FB15K237/rgcn_lp.pt` at epoch `45`   
step `0`, time `2021-08-24 18:56:20`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `46`, loss `0.15554073452949524`, time `2021-08-24 18:57:15`  
step `0`, time `2021-08-24 18:57:15`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `47`, loss `0.16111056506633759`, time `2021-08-24 18:58:11`  
step `0`, time `2021-08-24 18:58:11`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `48`, loss `0.1614076942205429`, time `2021-08-24 18:59:09`  
step `0`, time `2021-08-24 18:59:09`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `49`, loss `0.15011565387248993`, time `2021-08-24 19:00:01`  
step `0`, time `2021-08-24 19:00:01`   
number of entities: 14541, relations: 558771, edges: 558771   
- epoch `50`, loss `0.14717517793178558`, time `2021-08-24 19:00:56`  
