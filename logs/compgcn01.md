-----
### Running - `2021-10-19 21:43:37`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `200`
- number of negative triples: `1`
- learning rate: `0.001`
- dropout rate: `0.2`
- weight decay: `0.0`
- number of bases: `50`
- compgcn aggregation scheme: `add`
- number of subgraphs: `400`
- training cluster size: `12`
- number of epochs: `500`
- validation frequency: `1`
- training triple batch size: `128`
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
	 * number of triples in each cluster, min: 690, mean: 4174.264705882353, max: 7591
	 * loss `430.23065185546875`, time `2021-10-19 21:43:59`  
	 * validation results (raw)  at epoch `0`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.00722356466576457`  |  `0.006826736032962799`  |  `0.007025150582194328`  |  
		|  mean rank (MR)  |  `7225.90771484375`  |  `9304.1962890625`  |  `8265.0517578125`  |  
		|  mean equal (ME)  |  `61.08976364135742`  |  `6.753749847412109`  |  `33.921756744384766`  |  
		|  mrr considering equals  |  `0.007215105462819338`  |  `0.006826325785368681`  |  `0.007020715624094009`  |  
		|  mr considering equals  |  `7286.99755859375`  |  `9310.9501953125`  |  `8298.9736328125`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006501283147989735`  |  `0.006444254348445966`  |  `0.006472768748217851`  |  
		|  hits@10  |  `0.008839463929284289`  |  `0.008212147134302824`  |  `0.008525805531793556`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `0`   
* epoch 1
	 * number of triples in each cluster, min: 1749, mean: 4169.205882352941, max: 7877
	 * loss `555.3803100585938`, time `2021-10-19 21:46:40`  
	 * validation results (raw)  at epoch `1`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.00727213267236948`  |  `0.00676740612834692`  |  `0.0070197694003582`  |  
		|  mean rank (MR)  |  `5803.87744140625`  |  `8419.91796875`  |  `7111.8974609375`  |  
		|  mean equal (ME)  |  `49.715370178222656`  |  `10.196578025817871`  |  `29.955974578857422`  |  
		|  mrr considering equals  |  `0.007264740765094757`  |  `0.006766557693481445`  |  `0.007015649229288101`  |  
		|  mr considering equals  |  `5853.5927734375`  |  `8430.1142578125`  |  `7141.853515625`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.005189620758483034`  |  `0.00516110635871115`  |  
		|  hits@3  |  `0.005988023952095809`  |  `0.005873966353008269`  |  `0.005930995152552038`  |  
		|  hits@10  |  `0.008611348731109211`  |  `0.008212147134302824`  |  `0.008411747932706017`  |  
   
* epoch 2
	 * number of triples in each cluster, min: 2210, mean: 4191.676470588235, max: 7812
	 * loss `683.4916381835938`, time `2021-10-19 21:48:57`  
	 * validation results (raw)  at epoch `2`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.0072763110511004925`  |  `0.0068591577000916`  |  `0.0070677343755960464`  |  
		|  mean rank (MR)  |  `7274.02392578125`  |  `8851.0830078125`  |  `8062.5537109375`  |  
		|  mean equal (ME)  |  `49.605873107910156`  |  `7.745081424713135`  |  `28.675477981567383`  |  
		|  mrr considering equals  |  `0.007271054666489363`  |  `0.006858433596789837`  |  `0.007064743898808956`  |  
		|  mr considering equals  |  `7323.62939453125`  |  `8858.8271484375`  |  `8091.228515625`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006672369546621044`  |  `0.006102081551183348`  |  `0.006387225548902196`  |  
		|  hits@10  |  `0.008953521528371828`  |  `0.008212147134302824`  |  `0.008582834331337327`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `2`   
* epoch 3
	 * number of triples in each cluster, min: 906, mean: 4182.911764705882, max: 10470
	 * loss `716.7682495117188`, time `2021-10-19 21:51:28`  
	 * validation results (raw)  at epoch `3`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.007143033668398857`  |  `0.006709644105285406`  |  `0.006926339119672775`  |  
		|  mean rank (MR)  |  `7097.4208984375`  |  `8897.869140625`  |  `7997.64501953125`  |  
		|  mean equal (ME)  |  `42.41750717163086`  |  `4.520045757293701`  |  `23.46877670288086`  |  
		|  mrr considering equals  |  `0.007139300461858511`  |  `0.006709011271595955`  |  `0.006924156099557877`  |  
		|  mr considering equals  |  `7139.83837890625`  |  `8902.3896484375`  |  `8021.1142578125`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006045052751639578`  |  `0.005816937553464499`  |  `0.005930995152552038`  |  
		|  hits@10  |  `0.008782435129740519`  |  `0.008155118334759053`  |  `0.008468776732249786`  |  
   
* epoch 4
	 * number of triples in each cluster, min: 1028, mean: 4213.323529411765, max: 8185
	 * loss `467.67138671875`, time `2021-10-19 21:54:02`  
	 * validation results (raw)  at epoch `4`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.007215412799268961`  |  `0.006924299988895655`  |  `0.007069856394082308`  |  
		|  mean rank (MR)  |  `6895.42626953125`  |  `8809.58984375`  |  `7852.5078125`  |  
		|  mean equal (ME)  |  `31.81020736694336`  |  `5.420187950134277`  |  `18.615198135375977`  |  
		|  mrr considering equals  |  `0.007211628369987011`  |  `0.0069237700663506985`  |  `0.007067698985338211`  |  
		|  mr considering equals  |  `6927.23681640625`  |  `8815.009765625`  |  `7871.123046875`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006159110350727117`  |  `0.006273167949814656`  |  `0.006216139150270887`  |  
		|  hits@10  |  `0.00849729113202167`  |  `0.008326204733390363`  |  `0.008411747932706017`  |  
   
	 * model saved to `../pretrained/FB15K237/compgcn_lp.pt` at epoch `4`   
* epoch 5
	 * number of triples in each cluster, min: 1015, mean: 4163.970588235294, max: 7814
	 * loss `459.7915954589844`, time `2021-10-19 21:56:31`  
	 * validation results (raw)  at epoch `5`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.007084222976118326`  |  `0.006750253960490227`  |  `0.006917238235473633`  |  
		|  mean rank (MR)  |  `6666.90185546875`  |  `8733.201171875`  |  `7700.0517578125`  |  
		|  mean equal (ME)  |  `33.914798736572266`  |  `5.912574768066406`  |  `19.913686752319336`  |  
		|  mrr considering equals  |  `0.007081353571265936`  |  `0.00674971379339695`  |  `0.0069155339151620865`  |  
		|  mr considering equals  |  `6700.8173828125`  |  `8739.1142578125`  |  `7719.9658203125`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006159110350727117`  |  `0.006102081551183348`  |  `0.006130595950955232`  |  
		|  hits@10  |  `0.008383233532934131`  |  `0.008155118334759053`  |  `0.008269175933846592`  |  
   
* epoch 6
	 * number of triples in each cluster, min: 1228, mean: 4211.205882352941, max: 7998
	 * loss `522.6810913085938`, time `2021-10-19 21:59:05`  
	 * validation results (raw)  at epoch `6`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.0071091605350375175`  |  `0.006888365838676691`  |  `0.006998763419687748`  |  
		|  mean rank (MR)  |  `7064.12353515625`  |  `8558.26953125`  |  `7811.1962890625`  |  
		|  mean equal (ME)  |  `26.856685638427734`  |  `3.749016284942627`  |  `15.302850723266602`  |  
		|  mrr considering equals  |  `0.007107204291969538`  |  `0.006887934170663357`  |  `0.006997569464147091`  |  
		|  mr considering equals  |  `7090.98046875`  |  `8562.0185546875`  |  `7826.49951171875`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.006216139150270887`  |  `0.006501283147989735`  |  `0.006358711149130311`  |  
		|  hits@10  |  `0.008839463929284289`  |  `0.008098089535215285`  |  `0.008468776732249786`  |  
   
* epoch 7
	 * number of triples in each cluster, min: 1174, mean: 4215.14705882353, max: 7013
	 * loss `438.9543151855469`, time `2021-10-19 22:01:32`  
	 * validation results (raw)  at epoch `7`
   
		|  metric  |  head  |  tail  |  mean  |  
		|  ----  |  ----  |  ----  |  ----  |  
		|  mean reciprocal rank (MRR)  |  `0.006440899800509214`  |  `0.00630506407469511`  |  `0.006372981704771519`  |  
		|  mean rank (MR)  |  `5591.33349609375`  |  `6963.49658203125`  |  `6277.4150390625`  |  
		|  mean equal (ME)  |  `1916.017333984375`  |  `1007.2157592773438`  |  `1461.6165771484375`  |  
		|  mrr considering equals  |  `0.00641222158446908`  |  `0.006293922197073698`  |  `0.006353071890771389`  |  
		|  mr considering equals  |  `7507.3505859375`  |  `7970.712890625`  |  `7739.03173828125`  |  
		|  hits@1  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  `0.0051325919589392645`  |  
		|  hits@3  |  `0.005816937553464499`  |  `0.005873966353008269`  |  `0.005845451953236384`  |  
		|  hits@10  |  `0.007527801539777588`  |  `0.006957513544339892`  |  `0.0072426575420587395`  |  
   
* epoch 8
	 * number of triples in each cluster, min: 800, mean: 4174.088235294118, max: 6844
	 * loss `635.6203002929688`, time `2021-10-19 22:03:47`  
-----
### Running - `2021-10-19 22:04:32`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `200`
- number of negative triples: `1`
- learning rate: `0.001`
- dropout rate: `0.2`
- weight decay: `0.0`
- number of bases: `50`
- compgcn aggregation scheme: `add`
- number of subgraphs: `1`
- training cluster size: `12`
- number of epochs: `500`
- validation frequency: `1`
- training triple batch size: `128`
- validation/test triple batch size: `64`
- highest mrr: `0.0`
- device: `cpu`
- eval device: `cpu`
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
-----
### Running - `2021-10-19 22:07:30`
#### Configurations
- load data from `../data/FB15K237/`
- new training
- embedding dimension: `200`
- number of negative triples: `1`
- learning rate: `0.001`
- dropout rate: `0.2`
- weight decay: `0.0`
- number of bases: `50`
- compgcn aggregation scheme: `add`
- number of subgraphs: `1`
- training cluster size: `12`
- number of epochs: `500`
- validation frequency: `1`
- training triple batch size: `128`
- norm: `2`
- validation/test triple batch size: `64`
- highest mrr: `0.0`
- device: `cpu`
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
