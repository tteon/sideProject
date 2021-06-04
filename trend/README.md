# 0604 _ Fraud Detection ; Using Relational Graph Learning to Detect Collusion



source ; [uber page](https://eng.uber.com/fraud-detection/?utm_campaign=Learning%20Posts&utm_content=168312748&utm_medium=social&utm_source=twitter&hss_channel=tw-3018841323&s=09)





Financial criminals 에 대해 언급하며 특히 fraudulent behavior 에 대해 언급함. 예시를 들ㄷ자면 유저간의 collude 가 있다. 최종적으로 앞선 문제를 relational graph convolutional networks (RGCN)으로 해결해보고자 함.

Fraudulent(사기 유저) 는 종종 연걸되거나 군집된다, 

![Fraud Detection: Using Relational Graph Learning to Detect Collusion](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2021/05/Screen-Shot-2021-05-20-at-9.35.05-PM-768x552.png)

다음 figure1 은 그 상황에 대해 이해하기 쉽게 표현한 그림이다. 우리는 RGCN 모델을 활용하여 colluding user 와 다른 유저간의 관계를 활용한 case study 의 개요를 나타냄.
이 article 에 개발된 모델은 우버의 production platform 에서는 활용하고 있진 않음.

우리는 RGCN 모델을 데이터의 적은 량의 샘플데이터에 적용해보았다. 만약 유저가 사기에 관해 기여를 했을지에 대한 예측을 해보고자. 유저 그래프에는 2종류의 노드가 존재하는데 ‘운전자’와 ‘탑승자’가 존재함. ‘운전자’와 ‘탑승자’는 공유된 정보를 통해 연결될 수 있음. 그래프 내의 노드는 각각의 유저로 ㅎ확인할 수 있으며 , vector 의 임베딩으로 나타내진다. 그 representation 을 통하여 최종적으로 ml tasks에 쉽게 적용할 수 있음. 예를 들자면 node classification 이나 edge prediction 으로 ! 또한 우리는 단지 유저만의 특성을 활용하는게 아니라 주변 이웃(몇몇을 건너서까지)의 특성을 활용할 것임. 

Gcn 은 주변이웃들로부터의 정보를 학습하는데 유용하다는것을 보여준바 있따. 조금만 사족을 붙이자면 동일 웨이트를 전달하고자 엣지를 통해 source node에 도착한다. RGCN 은 이와 대조적으로 각각의 관계에 걸맞는 edge 즉, edge 마다의 weight가 달라 각 타입에 따라 가중치가 전달이 됨. 그러므로 각 노드는 edge type 의 정보에 따라 message 가 계산이 되어진다.

![img](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2021/05/Screen-Shot-2021-05-20-at-9.36.03-PM.png)

윗 그림은 RGCN 의 architecture 를 나타낸 것이다. 모델의 input은 node feature 그리고 edge type 으로 구성되어 있다. node feature 은 RGCN 레이어를 통해 변환되며 이 때 연결된 주변 이웃들의 노드들로부터 학습되어 aggreating 함. 이 때 학습에 쓰이는 message 는 각 edge type 에 따라 weight 가 다르다. 주변 이웃에 대한 정보를 받아드릴 때 weighted 그리고 normalized sum 된 값을 통한 값으로 종합되며 그것들을 적용한 값들을 토대로 target node 는 hidden representation 에 stacking 된다. 그 이후 activation function 인 ReLU 를 통해 활성 / 비활성 여부를 결정하게 됨. RGCN 레이어는 이 과정(convolution 그리고 message passing)을 통해 high-level node representation 을 추출할 수 있게 되어짐. 최종적으로 softmax layer 를 output layer 으로 적용하여 loss function cross-entropy 를 사용하여 해당 node score를 도출해냄.


$$
h^{(l+1)}_i = \sigma(\sum_{r\in R}\sum_{j\in N^r_i}\frac{1}{c_{i,r}}W^{(l)}_rh^{(l)_j}+W^{l}_0h^{l}_i)
$$




이웃 노드들의 변환된 feature vector은 엣지의 특성에 따라 좌지우지 된다. 즉 , edge 의 타입과 direction 에 따라 다르단 말이다. 

![img](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2021/05/Screen-Shot-2021-05-20-at-9.45.31-PM.png)

우버에는 fraudlent users 를 찾기위한 몇몇의 체크포인트와 다양한 risk model 이 존재한다. 그 중 risk model 을 통해 더욱 효율적으로 detection 하기 위해 고안한 아이디어는 fraud scores 를 가져온 다음 그것을 downstream risk model 의 feature 로 뢀용하는 것이다. 최종적으로 RGCN 에서 도출되는 값은 Fraud score이며 user의 리스크를 의미한다. 각각의 노드 hidden representation 는 user 가 fraudulent 인지 아닌지를 binary cross entropy loss 를 활용하여 학습한다(최소화하는 방향으로) . user 는 탑승자 그리고 운전ㅈㅏ 일 수도 있으므로 우리는 2가지 output (driver , rider)에서의 score 을 배출함.  

우리는 2가지 source 를 input 으로 사용했다. 노드피쳐 그리고 엣지타입. label 은 유저가 특정 time range 내에서의 지불거절의 여부였다. 우리는 model learning 에 대해 도움이 되고자 feature engineering 을 진행하였다. 예를 들자면 driver-rider 그래프는 2가지의 노드가 있을텐데 각각의 노드타입 즉, 드라이버 그리고 탑승자 는 다른 피쳐를 가질 수 있다. 그것을 풀기위해 우리는 zero padding 을 사용하여 input feature vectors 를 동일 길이를 갖게끔 조정해주었음. 두번째로 엣지타입에 대해 정의하고 또한 각각의 타입에 대해 weight를 지정해 주었다. 

RGCN model 의 performance 를 평가 및 fraud scores 의 활용을 하기 위해 우리는 historical data 를 4달 간격으로 split 하여 모델을 학습 시켰다.  그때 데이터에서 모델의 성능을 측정하기 위해 다음 6주간의 데이터를 예측하였다. 특별히 우리는 fraud score 를 output 하기 위해 precision ,recall, 그리고 auc 를 적용하였다. 실험동안 15% 정도 의 나은 precision  얻음을 관찰하였다 기존 production model 에 2가지 fraud score features 를 넣음으로써. 최종적으로 앞선 스코어들은 200가지의 feature 중 4th , 39th 중요성을 도출해냄.

