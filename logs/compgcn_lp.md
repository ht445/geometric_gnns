-----
### Running Time: `2021-06-24 19:28:03`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `10240`
- validation/test batch size: `1000`
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
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
#### training
- epoch `0`, loss `251.6344451904297`, time `2021-06-24 19:31:32`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `6534.701171875`  |  `9481.2080078125`  |  `8007.95458984375`  |  
|  mean reciprocal rank (MRR)  |  `0.07044374942779541`  |  `0.0069403876550495625`  |  `0.03869206830859184`  |  
|  hit@1  |  `0.0011666667414829135`  |  `0.0009444445022381842`  |  `0.0010555556509643793`  |  
|  hit@3  |  `0.13660694658756256`  |  `0.011041018180549145`  |  `0.07382398098707199`  |  
|  hit@10  |  `0.14120352268218994`  |  `0.013470926322042942`  |  `0.07733722776174545`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `199.80978393554688`, time `2021-06-24 19:38:04`  
- epoch `2`, loss `196.79664611816406`, time `2021-06-24 19:41:40`  
- epoch `3`, loss `193.26150512695312`, time `2021-06-24 19:45:17`  
- validation results  at epoch `3`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `7725.66064453125`  |  `10799.37890625`  |  `9262.51953125`  |  
|  mean reciprocal rank (MRR)  |  `0.07239696383476257`  |  `0.010943811386823654`  |  `0.04167038947343826`  |  
|  hit@1  |  `0.004929906222969294`  |  `0.005096573382616043`  |  `0.005013239569962025`  |  
|  hit@3  |  `0.13882189989089966`  |  `0.015470921993255615`  |  `0.07714641094207764`  |  
|  hit@10  |  `0.14032191038131714`  |  `0.0174080990254879`  |  `0.07886500656604767`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `3`   
- epoch `4`, loss `190.7243194580078`, time `2021-06-24 19:51:33`  
- epoch `5`, loss `186.65122985839844`, time `2021-06-24 19:55:04`  
-----
### Running Time: `2021-06-24 20:00:06`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `40960`
- validation/test batch size: `5000`
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
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
#### training
- epoch `0`, loss `-543.6928100585938`, time `2021-06-24 20:01:29`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `76.068115234375`  |  `109.3074722290039`  |  `92.68778991699219`  |  
|  mean reciprocal rank (MRR)  |  `0.35253071784973145`  |  `0.2577688694000244`  |  `0.30514979362487793`  |  
|  hit@1  |  `0.004933432210236788`  |  `0.004833431914448738`  |  `0.004883431829512119`  |  
|  hit@3  |  `0.6668194532394409`  |  `0.4479774236679077`  |  `0.5573984384536743`  |  
|  hit@10  |  `0.7840107679367065`  |  `0.639756441116333`  |  `0.7118836045265198`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `-657.3421020507812`, time `2021-06-24 20:05:00`  
- epoch `2`, loss `-658.6571655273438`, time `2021-06-24 20:06:27`  
- epoch `3`, loss `-658.7630004882812`, time `2021-06-24 20:07:50`  
- validation results  at epoch `3`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `2.333322286605835`  |  `2.3093807697296143`  |  `2.3213515281677246`  |  
|  mean reciprocal rank (MRR)  |  `0.5906513929367065`  |  `0.5892086029052734`  |  `0.58992999792099`  |  
|  hit@1  |  `0.2133861780166626`  |  `0.21348483860492706`  |  `0.21343550086021423`  |  
|  hit@3  |  `0.9631234407424927`  |  `0.9612908959388733`  |  `0.9622071981430054`  |  
|  hit@10  |  `0.9890875816345215`  |  `0.9914835691452026`  |  `0.9902855753898621`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `3`   
- epoch `4`, loss `-658.7776489257812`, time `2021-06-24 20:11:20`  
- epoch `5`, loss `-658.7874145507812`, time `2021-06-24 20:12:42`  
- epoch `6`, loss `-658.790283203125`, time `2021-06-24 20:14:04`  
- validation results  at epoch `6`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1.844467282295227`  |  `1.8089845180511475`  |  `1.826725959777832`  |  
|  mean reciprocal rank (MRR)  |  `0.6901900172233582`  |  `0.6926900744438171`  |  `0.6914400458335876`  |  
|  hit@1  |  `0.4010666608810425`  |  `0.4013640284538269`  |  `0.4012153446674347`  |  
|  hit@3  |  `0.9761278629302979`  |  `0.9841570258140564`  |  `0.9801424741744995`  |  
|  hit@10  |  `0.9944698810577393`  |  `0.9974579811096191`  |  `0.9959639310836792`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `6`   
-----
### Running Time: `2021-06-24 20:26:37`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `20480`
- validation/test batch size: `2000`
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
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
#### training
- epoch `0`, loss `18.73331642150879`, time `2021-06-24 20:28:40`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5328.97509765625`  |  `6805.84521484375`  |  `6067.41015625`  |  
|  mean reciprocal rank (MRR)  |  `0.07304186373949051`  |  `0.012456972151994705`  |  `0.042749419808387756`  |  
|  hit@1  |  `0.0031397033017128706`  |  `0.0019225480500608683`  |  `0.0025311256758868694`  |  
|  hit@3  |  `0.137791708111763`  |  `0.013023525476455688`  |  `0.07540761679410934`  |  
|  hit@10  |  `0.14550542831420898`  |  `0.02816648967564106`  |  `0.08683595806360245`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `13.883780479431152`, time `2021-06-24 20:33:01`  
- epoch `2`, loss `13.586288452148438`, time `2021-06-24 20:35:08`  
- epoch `3`, loss `13.332696914672852`, time `2021-06-24 20:37:09`  
- validation results  at epoch `3`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `2467.45361328125`  |  `4066.084228515625`  |  `3266.76904296875`  |  
|  mean reciprocal rank (MRR)  |  `0.08108765631914139`  |  `0.03211488202214241`  |  `0.05660127103328705`  |  
|  hit@1  |  `0.009329895488917828`  |  `0.01081813219934702`  |  `0.010074013844132423`  |  
|  hit@3  |  `0.13940788805484772`  |  `0.03338364139199257`  |  `0.086395762860775`  |  
|  hit@10  |  `0.16038073599338531`  |  `0.07033786177635193`  |  `0.11535929888486862`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `3`   
- epoch `4`, loss `13.08562183380127`, time `2021-06-24 20:41:27`  
- epoch `5`, loss `12.695109367370605`, time `2021-06-24 20:43:30`  
- epoch `6`, loss `12.834386825561523`, time `2021-06-24 20:45:38`  
- validation results  at epoch `6`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1228.157958984375`  |  `3027.656982421875`  |  `2127.907470703125`  |  
|  mean reciprocal rank (MRR)  |  `0.1073971688747406`  |  `0.06423494219779968`  |  `0.08581605553627014`  |  
|  hit@1  |  `0.0193550493568182`  |  `0.027612559497356415`  |  `0.02348380535840988`  |  
|  hit@3  |  `0.17214496433734894`  |  `0.07006008177995682`  |  `0.12110252678394318`  |  
|  hit@10  |  `0.2122775912284851`  |  `0.13329008221626282`  |  `0.17278383672237396`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `6`   
- epoch `7`, loss `12.758902549743652`, time `2021-06-24 20:50:00`  
- epoch `8`, loss `12.745312690734863`, time `2021-06-24 20:52:07`  
- epoch `9`, loss `12.67426872253418`, time `2021-06-24 20:54:13`  
- validation results  at epoch `9`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `859.2581176757812`  |  `2506.919189453125`  |  `1683.088623046875`  |  
|  mean reciprocal rank (MRR)  |  `0.1134580671787262`  |  `0.08578705787658691`  |  `0.09962256252765656`  |  
|  hit@1  |  `0.022176619619131088`  |  `0.041193265467882156`  |  `0.03168494254350662`  |  
|  hit@3  |  `0.17847321927547455`  |  `0.09381739795207977`  |  `0.13614530861377716`  |  
|  hit@10  |  `0.22272872924804688`  |  `0.17395150661468506`  |  `0.19834011793136597`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `9`   
- epoch `10`, loss `12.559708595275879`, time `2021-06-24 20:58:31`  
- epoch `11`, loss `12.55249309539795`, time `2021-06-24 21:00:37`  
- epoch `12`, loss `12.514989852905273`, time `2021-06-24 21:02:45`  
- validation results  at epoch `12`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `752.1272583007812`  |  `2310.205322265625`  |  `1531.166259765625`  |  
|  mean reciprocal rank (MRR)  |  `0.11314621567726135`  |  `0.09704235196113586`  |  `0.10509428381919861`  |  
|  hit@1  |  `0.024673178791999817`  |  `0.05158378183841705`  |  `0.038128480315208435`  |  
|  hit@3  |  `0.16695477068424225`  |  `0.10480383038520813`  |  `0.1358793079853058`  |  
|  hit@10  |  `0.23421335220336914`  |  `0.18446651101112366`  |  `0.2093399316072464`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `12`   
- epoch `13`, loss `12.517437934875488`, time `2021-06-24 21:07:10`  
- epoch `14`, loss `12.538474082946777`, time `2021-06-24 21:09:10`  
- epoch `15`, loss `12.505240440368652`, time `2021-06-24 21:11:15`  
- validation results  at epoch `15`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `729.3411865234375`  |  `2304.6142578125`  |  `1516.977783203125`  |  
|  mean reciprocal rank (MRR)  |  `0.1129235178232193`  |  `0.09922666847705841`  |  `0.10607509315013885`  |  
|  hit@1  |  `0.023893777281045914`  |  `0.05223018676042557`  |  `0.03806198388338089`  |  
|  hit@3  |  `0.16569381952285767`  |  `0.10683587193489075`  |  `0.1362648457288742`  |  
|  hit@10  |  `0.23505012691020966`  |  `0.1908014863729477`  |  `0.21292580664157867`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `15`   
- epoch `16`, loss `12.483860969543457`, time `2021-06-24 21:15:36`  
-----
### Running Time: `2021-06-24 21:18:16`
#### configuration
- load data from `../data/FB15K237/`
- operation: `RGCN`
- continue training: `False`
- embedding dimension: `200`
- number of bases: `100`
- compGCN aggregation scheme: `add`
- train batch size: `40960`
- validation/test batch size: `1000`
- learning rate: `0.01`
- number of epochs: `100`
- number of negative triples: `32`
- number of entities: `14541`
- number of relations: `475`
- number of training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
- patience: `2`
- norm: `2`
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_r_weight', 'compgcn.relation_l_weight']`
-----
### Running Time: `2021-06-24 21:31:56`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `200`
- number of bases: `100`
- compGCN aggregation scheme: `add`
- train batch size: `10240`
- validation/test batch size: `500`
- learning rate: `0.01`
- number of epochs: `100`
- number of negative triples: `32`
- number of entities: `14541`
- number of relations: `475`
- number of training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
- patience: `2`
- norm: `2`
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
-----
### Running Time: `2021-06-24 21:44:57`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `200`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `10240`
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
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
-----
### Running Time: `2021-06-24 21:57:22`
#### configuration
- load data from `../data/FB15K237/`
- operation: `TransE`
- continue training: `False`
- embedding dimension: `100`
- number of bases: `50`
- compGCN aggregation scheme: `add`
- train batch size: `20480`
- validation/test batch size: `2000`
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
- CompGCN instantiated
- model parameters: `['entity_embeds', 'compgcn.bases', 'compgcn.coefficients', 'compgcn.weights', 'compgcn.relation_weight']`
#### training
- epoch `0`, loss `18.144447326660156`, time `2021-06-24 21:59:30`  
- validation results  at epoch `0`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `5178.70751953125`  |  `6654.494140625`  |  `5916.6005859375`  |  
|  mean reciprocal rank (MRR)  |  `0.07975306361913681`  |  `0.014275316148996353`  |  `0.04701419174671173`  |  
|  hit@1  |  `0.002528592012822628`  |  `0.00269525870680809`  |  `0.002611925359815359`  |  
|  hit@3  |  `0.15026132762432098`  |  `0.016041982918977737`  |  `0.08315165340900421`  |  
|  hit@10  |  `0.16127470135688782`  |  `0.025538545101881027`  |  `0.09340662509202957`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
- epoch `1`, loss `15.25090503692627`, time `2021-06-24 22:03:53`  
- epoch `2`, loss `14.727387428283691`, time `2021-06-24 22:05:58`  
- epoch `3`, loss `14.081427574157715`, time `2021-06-24 22:08:06`  
- validation results  at epoch `3`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `2248.37158203125`  |  `4104.5361328125`  |  `3176.453857421875`  |  
|  mean reciprocal rank (MRR)  |  `0.08607448637485504`  |  `0.034074895083904266`  |  `0.060074690729379654`  |  
|  hit@1  |  `0.007834961637854576`  |  `0.008935938589274883`  |  `0.008385449647903442`  |  
|  hit@3  |  `0.14833714067935944`  |  `0.037213537842035294`  |  `0.09277533739805222`  |  
|  hit@10  |  `0.1829681396484375`  |  `0.08405157923698425`  |  `0.13350985944271088`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `3`   
- epoch `4`, loss `13.786099433898926`, time `2021-06-24 22:12:32`  
- epoch `5`, loss `13.345638275146484`, time `2021-06-24 22:14:41`  
- epoch `6`, loss `13.625468254089355`, time `2021-06-24 22:16:51`  
- validation results  at epoch `6`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `1215.068359375`  |  `3036.739501953125`  |  `2125.90380859375`  |  
|  mean reciprocal rank (MRR)  |  `0.10677794367074966`  |  `0.05539652332663536`  |  `0.08108723163604736`  |  
|  hit@1  |  `0.016848353669047356`  |  `0.02212250977754593`  |  `0.019485432654619217`  |  
|  hit@3  |  `0.17563825845718384`  |  `0.06087718904018402`  |  `0.11825772374868393`  |  
|  hit@10  |  `0.2098972201347351`  |  `0.11888465285301208`  |  `0.1643909364938736`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `6`   
- epoch `7`, loss `13.49409008026123`, time `2021-06-24 22:21:31`  
- epoch `8`, loss `13.361564636230469`, time `2021-06-24 22:23:39`  
- epoch `9`, loss `13.2092924118042`, time `2021-06-24 22:25:49`  
- validation results  at epoch `9`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `871.089111328125`  |  `2555.594482421875`  |  `1713.341796875`  |  
|  mean reciprocal rank (MRR)  |  `0.11391157656908035`  |  `0.07382161915302277`  |  `0.09386660158634186`  |  
|  hit@1  |  `0.01963789574801922`  |  `0.03001997247338295`  |  `0.02482893317937851`  |  
|  hit@3  |  `0.18131324648857117`  |  `0.08285273611545563`  |  `0.132082998752594`  |  
|  hit@10  |  `0.22766123712062836`  |  `0.15572819113731384`  |  `0.1916947066783905`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `9`   
- epoch `10`, loss `12.985733985900879`, time `2021-06-24 22:30:07`  
- epoch `11`, loss `12.887588500976562`, time `2021-06-24 22:32:40`  
- epoch `12`, loss `13.055397033691406`, time `2021-06-24 22:34:47`  
- validation results  at epoch `12`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `761.2006225585938`  |  `2396.0791015625`  |  `1578.639892578125`  |  
|  mean reciprocal rank (MRR)  |  `0.12634915113449097`  |  `0.09430703520774841`  |  `0.11032809317111969`  |  
|  hit@1  |  `0.030203402042388916`  |  `0.045460913330316544`  |  `0.03783215582370758`  |  
|  hit@3  |  `0.19455556571483612`  |  `0.10561871528625488`  |  `0.1500871479511261`  |  
|  hit@10  |  `0.24328067898750305`  |  `0.1895068734884262`  |  `0.21639376878738403`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `12`   
- epoch `13`, loss `13.189995765686035`, time `2021-06-24 22:39:09`  
- epoch `14`, loss `13.167512893676758`, time `2021-06-24 22:41:43`  
- epoch `15`, loss `13.173493385314941`, time `2021-06-24 22:44:26`  
- validation results  at epoch `15`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `720.991455078125`  |  `2354.291748046875`  |  `1537.6416015625`  |  
|  mean reciprocal rank (MRR)  |  `0.12438318133354187`  |  `0.09144166857004166`  |  `0.10791242122650146`  |  
|  hit@1  |  `0.027274157851934433`  |  `0.04690191522240639`  |  `0.03708803653717041`  |  
|  hit@3  |  `0.19114476442337036`  |  `0.09818440675735474`  |  `0.14466458559036255`  |  
|  hit@10  |  `0.24115273356437683`  |  `0.1748858094215393`  |  `0.20801927149295807`  |  
   
- epoch `16`, loss `13.128890037536621`, time `2021-06-24 22:49:33`  
- epoch `17`, loss `13.124356269836426`, time `2021-06-24 22:52:05`  
- epoch `18`, loss `13.135824203491211`, time `2021-06-24 22:54:41`  
- validation results  at epoch `18`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `702.1651000976562`  |  `2312.25341796875`  |  `1507.209228515625`  |  
|  mean reciprocal rank (MRR)  |  `0.1257762908935547`  |  `0.10071732103824615`  |  `0.11324680596590042`  |  
|  hit@1  |  `0.02710242196917534`  |  `0.048574913293123245`  |  `0.03783866763114929`  |  
|  hit@3  |  `0.19383332133293152`  |  `0.11103769391775131`  |  `0.1524355113506317`  |  
|  hit@10  |  `0.24616289138793945`  |  `0.20491613447666168`  |  `0.22553950548171997`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `18`   
- epoch `19`, loss `12.95264720916748`, time `2021-06-24 22:59:33`  
- epoch `20`, loss `12.881528854370117`, time `2021-06-24 23:01:40`  
- epoch `21`, loss `12.845953941345215`, time `2021-06-24 23:03:49`  
- validation results  at epoch `21`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `671.0429077148438`  |  `2165.0712890625`  |  `1418.05712890625`  |  
|  mean reciprocal rank (MRR)  |  `0.13083001971244812`  |  `0.1054585799574852`  |  `0.11814430356025696`  |  
|  hit@1  |  `0.0484507791697979`  |  `0.057809267193078995`  |  `0.053130023181438446`  |  
|  hit@3  |  `0.1871313601732254`  |  `0.11730203032493591`  |  `0.15221670269966125`  |  
|  hit@10  |  `0.2413565069437027`  |  `0.19088564813137054`  |  `0.21612107753753662`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `21`   
- epoch `22`, loss `12.830811500549316`, time `2021-06-24 23:08:06`  
- epoch `23`, loss `12.83022689819336`, time `2021-06-24 23:10:16`  
- epoch `24`, loss `12.774346351623535`, time `2021-06-24 23:12:26`  
- validation results  at epoch `24`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `638.742919921875`  |  `2183.97119140625`  |  `1411.3570556640625`  |  
|  mean reciprocal rank (MRR)  |  `0.1259182244539261`  |  `0.10996634513139725`  |  `0.11794228851795197`  |  
|  hit@1  |  `0.02995765581727028`  |  `0.05868639051914215`  |  `0.044322021305561066`  |  
|  hit@3  |  `0.19421714544296265`  |  `0.12282899022102356`  |  `0.1585230678319931`  |  
|  hit@10  |  `0.2597956955432892`  |  `0.20521551370620728`  |  `0.23250560462474823`  |  
   
- epoch `25`, loss `12.78407096862793`, time `2021-06-24 23:17:27`  
- epoch `26`, loss `12.78698444366455`, time `2021-06-24 23:19:52`  
- epoch `27`, loss `12.801250457763672`, time `2021-06-24 23:22:25`  
- validation results  at epoch `27`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `625.140869140625`  |  `2204.66015625`  |  `1414.9005126953125`  |  
|  mean reciprocal rank (MRR)  |  `0.11810917407274246`  |  `0.11273074150085449`  |  `0.11541995406150818`  |  
|  hit@1  |  `0.027378574013710022`  |  `0.05934121832251549`  |  `0.043359898030757904`  |  
|  hit@3  |  `0.1870303899049759`  |  `0.1283227801322937`  |  `0.1576765775680542`  |  
|  hit@10  |  `0.2475028783082962`  |  `0.21481481194496155`  |  `0.23115885257720947`  |  
   
#### testing
- CompGCN instantiated
- testing results  
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `673.3828735351562`  |  `2129.835693359375`  |  `1401.6092529296875`  |  
|  mean reciprocal rank (MRR)  |  `0.12702743709087372`  |  `0.10114094614982605`  |  `0.11408419162034988`  |  
|  hit@1  |  `0.0467463880777359`  |  `0.05306652560830116`  |  `0.04990645498037338`  |  
|  hit@3  |  `0.18180181086063385`  |  `0.11458759009838104`  |  `0.14819470047950745`  |  
|  hit@10  |  `0.23247618973255157`  |  `0.1872071623802185`  |  `0.20984166860580444`  |  
   
-----
  
