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
- epoch `7`, loss `1.0607800483703613`, time `2021-06-25 00:13:05`  
- epoch `8`, loss `1.006730318069458`, time `2021-06-25 00:17:47`  
- epoch `9`, loss `0.9528452157974243`, time `2021-06-25 00:22:25`  
- validation results  at epoch `9`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `442.62982177734375`  |  `238.17315673828125`  |  `340.4014892578125`  |  
|  mean reciprocal rank (MRR)  |  `0.0823899656534195`  |  `0.21951749920845032`  |  `0.1509537398815155`  |  
|  hit@1  |  `0.03689111769199371`  |  `0.1382104754447937`  |  `0.08755079656839371`  |  
|  hit@3  |  `0.07840977609157562`  |  `0.2415008842945099`  |  `0.15995532274246216`  |  
|  hit@10  |  `0.17247086763381958`  |  `0.378403902053833`  |  `0.2754373848438263`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `9`   
- epoch `10`, loss `0.9237700700759888`, time `2021-06-25 00:44:06`  
- epoch `11`, loss `0.9007005095481873`, time `2021-06-25 00:48:48`  
- epoch `12`, loss `0.848780632019043`, time `2021-06-25 00:53:33`  
- validation results  at epoch `12`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `401.3372497558594`  |  `208.6848907470703`  |  `305.0110778808594`  |  
|  mean reciprocal rank (MRR)  |  `0.08814077824354172`  |  `0.2289307862520218`  |  `0.15853577852249146`  |  
|  hit@1  |  `0.041073884814977646`  |  `0.14546920359134674`  |  `0.09327154606580734`  |  
|  hit@3  |  `0.08420276641845703`  |  `0.24999472498893738`  |  `0.1670987457036972`  |  
|  hit@10  |  `0.1810489296913147`  |  `0.3982701301574707`  |  `0.2896595299243927`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `12`   
- epoch `13`, loss `0.8125735521316528`, time `2021-06-25 01:15:19`  
- epoch `14`, loss `0.822592556476593`, time `2021-06-25 01:20:01`  
- epoch `15`, loss `0.8423323631286621`, time `2021-06-25 01:25:03`  
- validation results  at epoch `15`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `389.4001770019531`  |  `201.25869750976562`  |  `295.3294372558594`  |  
|  mean reciprocal rank (MRR)  |  `0.0922907292842865`  |  `0.2313476949930191`  |  `0.1618192195892334`  |  
|  hit@1  |  `0.04200713708996773`  |  `0.145542174577713`  |  `0.09377465397119522`  |  
|  hit@3  |  `0.08960212767124176`  |  `0.25228801369667053`  |  `0.17094507813453674`  |  
|  hit@10  |  `0.1914479285478592`  |  `0.4038203954696655`  |  `0.29763415455818176`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `15`   
- epoch `16`, loss `0.7761903405189514`, time `2021-06-25 01:45:32`  
- epoch `17`, loss `0.7436051964759827`, time `2021-06-25 01:53:12`  
- epoch `18`, loss `0.7198050618171692`, time `2021-06-25 01:57:37`  
- validation results  at epoch `18`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `361.96173095703125`  |  `183.0995330810547`  |  `272.5306396484375`  |  
|  mean reciprocal rank (MRR)  |  `0.09691934287548065`  |  `0.23652230203151703`  |  `0.16672082245349884`  |  
|  hit@1  |  `0.045129433274269104`  |  `0.14753152430057526`  |  `0.09633047878742218`  |  
|  hit@3  |  `0.09309208393096924`  |  `0.25624746084213257`  |  `0.1746697723865509`  |  
|  hit@10  |  `0.2020190805196762`  |  `0.42033469676971436`  |  `0.3111768960952759`  |  
   
- model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `18`   
- epoch `19`, loss `0.710798978805542`, time `2021-06-25 02:17:52`  
- epoch `20`, loss `0.7016952037811279`, time `2021-06-25 02:22:13`  
- epoch `21`, loss `0.6845260858535767`, time `2021-06-25 02:26:41`  
- validation results  at epoch `21`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `355.4068908691406`  |  `181.70455932617188`  |  `268.55572509765625`  |  
|  mean reciprocal rank (MRR)  |  `0.09736278653144836`  |  `0.2208409607410431`  |  `0.15910187363624573`  |  
|  hit@1  |  `0.04363579303026199`  |  `0.13331405818462372`  |  `0.08847492933273315`  |  
|  hit@3  |  `0.09530176222324371`  |  `0.23850832879543304`  |  `0.16690504550933838`  |  
|  hit@10  |  `0.20207536220550537`  |  `0.4023641347885132`  |  `0.3022197484970093`  |  
   
- epoch `22`, loss `0.6866558790206909`, time `2021-06-25 02:46:28`  
- epoch `23`, loss `0.6578766107559204`, time `2021-06-25 02:50:55`  
- epoch `24`, loss `0.672065258026123`, time `2021-06-25 02:55:19`  
- validation results  at epoch `24`
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `362.97216796875`  |  `186.7592315673828`  |  `274.8656921386719`  |  
|  mean reciprocal rank (MRR)  |  `0.09956229478120804`  |  `0.22360049188137054`  |  `0.1615813970565796`  |  
|  hit@1  |  `0.04479365423321724`  |  `0.13450513780117035`  |  `0.08964939415454865`  |  
|  hit@3  |  `0.0977698415517807`  |  `0.2429116815328598`  |  `0.17034076154232025`  |  
|  hit@10  |  `0.2102460116147995`  |  `0.40934842824935913`  |  `0.3097972273826599`  |  
   
#### testing
- CompRGCN instantiated
- testing results  
   
|  metric  |  head  |  tail  |  mean  |  
|  ----  |  ----  |  ----  |  ----  |  
|  mean rank (MR)  |  `358.2768859863281`  |  `185.40789794921875`  |  `271.8424072265625`  |  
|  mean reciprocal rank (MRR)  |  `0.09944625943899155`  |  `0.23323273658752441`  |  `0.16633950173854828`  |  
|  hit@1  |  `0.04600272700190544`  |  `0.14427782595157623`  |  `0.09514027833938599`  |  
|  hit@3  |  `0.09846094995737076`  |  `0.25662872195243835`  |  `0.17754483222961426`  |  
|  hit@10  |  `0.20743079483509064`  |  `0.41294726729393005`  |  `0.31018903851509094`  |  
   
-----
  
