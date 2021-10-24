
-----
### Running - `2021-10-20 17:50:10`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `200`
- number of negative triples: `1`
- learning rate: `0.001`
- dropout rate: `0.2`
- number of bases: `50`
- compgcn aggregation scheme: `add`
- number of subgraphs: `400`
- training cluster size: `12`
- number of epochs: `50`
- validation frequency: `1`
- training triple batch size: `128`
- norm: `2`
- margin: `1`
- validation/test triple batch size: `64`
- highest mrr: `0.0`
- device: `cuda:2`
- eval device: `cuda:3`
#### Preparing Data
- number of entities: `14541`
- number of original relations: `237`
- number of original training triples: `272115`
- number of validation triples: `17535`
- number of testing triples: `20466`
Computing METIS partitioning...
Done!
#### Model Training and Validation
* epoch 0
	 * number of triples in each cluster, min: 777, mean: 4193.441176470588, max: 8618
	 * loss `140.20419311523438`, time `2021-10-20 17:50:33`  
	 * validation results (raw)  at epoch `0`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.009682869538664818`  |  `0.009183140471577644`  |  `0.009433005005121231`  |  
		|  mean rank (MR)  |  `2509.276123046875`  |  `2345.266845703125`  |  `2427.271484375`  |  
		|  mean equal (ME)  |  `0.022811518982052803`  |  `0.06649558246135712`  |  `0.044653549790382385`  |  
		|  mrr considering equals  |  `0.009665533900260925`  |  `0.009169945493340492`  |  `0.009417739696800709`  |  
		|  mr considering equals  |  `2509.298583984375`  |  `2345.33349609375`  |  `2427.31591796875`  |  
		|  hits@1  |  `0.004619332763045338`  |  `0.004505275163957798`  |  `0.004562303963501569`  |  
		|  hits@3  |  `0.0072426575420587395`  |  `0.006045052751639578`  |  `0.006643855146849158`  |  
		|  hits@10  |  `0.013515825491873397`  |  `0.010778443113772455`  |  `0.012147134302822927`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
* epoch 1
	 * number of triples in each cluster, min: 1969, mean: 4201.382352941177, max: 7433
	 * loss `72.03659057617188`, time `2021-10-20 17:52:05`  
	 * validation results (raw)  at epoch `1`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.010261018760502338`  |  `0.035087477415800095`  |  `0.02267424762248993`  |  
		|  mean rank (MR)  |  `2048.060546875`  |  `1343.1025390625`  |  `1695.58154296875`  |  
		|  mean equal (ME)  |  `0.07738807797431946`  |  `0.7162817120552063`  |  `0.3968349099159241`  |  
		|  mrr considering equals  |  `0.010258486494421959`  |  `0.03491949662566185`  |  `0.02258899062871933`  |  
		|  mr considering equals  |  `2048.1376953125`  |  `1343.81884765625`  |  `1695.978271484375`  |  
		|  hits@1  |  `0.003763900769888794`  |  `0.01842030225263758`  |  `0.011092101511263188`  |  
		|  hits@3  |  `0.006045052751639578`  |  `0.027830054177359568`  |  `0.016937553464499572`  |  
		|  hits@10  |  `0.01454234388366125`  |  `0.05765611633875107`  |  `0.03609923011120616`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `1`   
* epoch 2
	 * number of triples in each cluster, min: 896, mean: 4192.14705882353, max: 6847
	 * loss `56.250240325927734`, time `2021-10-20 17:53:34`  
	 * validation results (raw)  at epoch `2`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.014806822873651981`  |  `0.03695984184741974`  |  `0.025883331894874573`  |  
		|  mean rank (MR)  |  `1862.1669921875`  |  `1208.2034912109375`  |  `1535.185302734375`  |  
		|  mean equal (ME)  |  `0.03855146840214729`  |  `0.07140005379915237`  |  `0.05497576296329498`  |  
		|  mrr considering equals  |  `0.014760809950530529`  |  `0.036897704005241394`  |  `0.02582925744354725`  |  
		|  mr considering equals  |  `1862.20556640625`  |  `1208.27490234375`  |  `1535.240234375`  |  
		|  hits@1  |  `0.005930995152552038`  |  `0.015968063872255488`  |  `0.010949529512403763`  |  
		|  hits@3  |  `0.009751924721984603`  |  `0.031080695751354435`  |  `0.020416310236669517`  |  
		|  hits@10  |  `0.024522383803820928`  |  `0.06780724265754205`  |  `0.04616481323068149`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `2`   
* epoch 3
	 * number of triples in each cluster, min: 904, mean: 4195.14705882353, max: 6976
	 * loss `49.10811996459961`, time `2021-10-20 17:55:05`  
	 * validation results (raw)  at epoch `3`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.014451746828854084`  |  `0.03334013372659683`  |  `0.02389593981206417`  |  
		|  mean rank (MR)  |  `1844.2064208984375`  |  `1148.3048095703125`  |  `1496.255615234375`  |  
		|  mean equal (ME)  |  `0.09831765294075012`  |  `0.12711718678474426`  |  `0.11271741986274719`  |  
		|  mrr considering equals  |  `0.01420868281275034`  |  `0.03323325887322426`  |  `0.02372097037732601`  |  
		|  mr considering equals  |  `1844.3048095703125`  |  `1148.4317626953125`  |  `1496.3682861328125`  |  
		|  hits@1  |  `0.004961505560307955`  |  `0.015112631879098944`  |  `0.01003706871970345`  |  
		|  hits@3  |  `0.009067579127459367`  |  `0.03051040775591674`  |  `0.019788993441688052`  |  
		|  hits@10  |  `0.025206729398346166`  |  `0.05343598517251212`  |  `0.03932135728542914`  |  
   
* epoch 4
	 * number of triples in each cluster, min: 1732, mean: 4180.205882352941, max: 8187
	 * loss `43.56689453125`, time `2021-10-20 17:56:36`  
	 * validation results (raw)  at epoch `4`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.01408313401043415`  |  `0.03512942045927048`  |  `0.02460627630352974`  |  
		|  mean rank (MR)  |  `1653.153076171875`  |  `836.088623046875`  |  `1244.620849609375`  |  
		|  mean equal (ME)  |  `0.055831193923950195`  |  `0.15831194818019867`  |  `0.10707157105207443`  |  
		|  mrr considering equals  |  `0.014053918421268463`  |  `0.03508467972278595`  |  `0.024569299072027206`  |  
		|  mr considering equals  |  `1653.2088623046875`  |  `836.2469482421875`  |  `1244.7279052734375`  |  
		|  hits@1  |  `0.003992015968063872`  |  `0.010322212717422298`  |  `0.007157114342743085`  |  
		|  hits@3  |  `0.00872540633019675`  |  `0.028457370972341033`  |  `0.018591388651268893`  |  
		|  hits@10  |  `0.024693470202452238`  |  `0.07082976903336184`  |  `0.047761619617907036`  |  
   
* epoch 5
	 * number of triples in each cluster, min: 986, mean: 4188.264705882353, max: 8357
	 * loss `39.950355529785156`, time `2021-10-20 17:58:04`  
	 * validation results (raw)  at epoch `5`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.016662491485476494`  |  `0.041541688144207`  |  `0.029102090746164322`  |  
		|  mean rank (MR)  |  `1436.7967529296875`  |  `842.3016357421875`  |  `1139.5491943359375`  |  
		|  mean equal (ME)  |  `0.009409751743078232`  |  `0.10881094634532928`  |  `0.05911035090684891`  |  
		|  mrr considering equals  |  `0.01662534661591053`  |  `0.041537970304489136`  |  `0.02908165752887726`  |  
		|  mr considering equals  |  `1436.80615234375`  |  `842.4104614257812`  |  `1139.6082763671875`  |  
		|  hits@1  |  `0.0053607071571143425`  |  `0.0213857998289136`  |  `0.013373253493013972`  |  
		|  hits@3  |  `0.010550327915597377`  |  `0.0388366124893071`  |  `0.024693470202452238`  |  
		|  hits@10  |  `0.028514399771884802`  |  `0.06951810664385515`  |  `0.049016253207869974`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `5`   
* epoch 6
	 * number of triples in each cluster, min: 2060, mean: 4165.264705882353, max: 7561
	 * loss `36.13058853149414`, time `2021-10-20 17:59:34`  

