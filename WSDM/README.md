# `DGL` tutorial _ wsdm



Node Classification?



# Intro



## Shallow embedding Vs. Graph neural network



Before Graph neural networks , **many proposed methods are using either connectivity alone ( such as DeepWalk or Node2vec), or simple combinations of connectivity and the node's own features.**

**GNNs, by contrast, offers an opportunity to obtain node representations by combining the connectivity and features of a local neighborhood.**

Shallow embedding은 단순 연결관계 혹은 자기 자신의 노드의 피쳐만을 사용하여 embedding을 한다. 허나 GNN은 이와 대조적으로 연결된 주변이웃의 feature들도 aggregation하면서 좀 더 다양한 feature을 얻게된다.



## Data splitting

늘 헷갈렸던 부분인데 DGL에서는 train val test 를 따로 splitting 을 해주고자 masking을 해준다. 

- train_mask ; A boolean tensor (True , False) 으로 노드가 training set 인지 아닌지 지칭한다.
- val_mask ; A boolean tensor (True , False) 으로 노드가 training set 인지 아닌지 지칭한다.
- test_mask ; A boolean tensor (True , False) 으로 노드가 training set 인지 아닌지 지칭한다.
- label ; 노드 카테고리에 대한 정답
- feat ; 노드 피쳐



## Modeling ( Defining a Graph Convolutional Network)



간단하다. torch.nn.Module 로 부터 상속받아 dgl.nn.GraphConv 모듈을 사용하면 된다.

이 때 layer을 쌓는다는 의미는 hop과도 같다 즉, 2개의 layer를 쌓는다는건 2다리를 걸친 이웃의 connectivity 와 feature을 가져온다는 것을 의미함.

Tips. **주변 이웃들을 어떻게 aggregation 하는가에 대한 다양한 알고리즘이 존재하는데 이를 DGL에서는 간단한 2~3라인의 코드로 load 할 수 있다.**



## Training the GCN

training 하는것은 기존 딥러닝프레임워크와 별반 차이없다. 다음과 같다

```python
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    for e in range(100):
        # Forward
        logits = model(g, features)
        
        # Compute prediction
        pred = logits.argmax(1)
        
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.corss_entropy(logits[train_mask], labels[train_mask])
        
        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            # 늘 헷갈렸던 f'' 문법 그냥 하면 된다!
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc)),
            print(f'In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} ,best {best_test_acc:.3f}')

# input -> hidden layer -> output
model = GCN(g.ndata['feat'].shape[1], 16, dataset_num_classes)
train(g, model)

# 이 때 유념해야할 것은 앞선 코드들은 cpu에서 작동하는 것임 만일 gpu를 활용하기 위해서는

1. g.to('cuda') 로 그래프 데이터를 tensor화 해준다.
2. model = GCN(~~~).to('cuda')
3. train

데이터 , 모델을 모두 to('cuda')로 텐서화 시켜줘야함.       
        
        
```



**만약 data가 unbalanced 하다면 Evaluation metric 을 accuracy 가 아닌 F1score를 활용해아한다. 그에 대해 DGL은 pytorch 기반으로 제작된 프레임워크이기에 활용할 수 있는 자료는 다음과 같다. https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354**



# Graph schema construction



DGL 은 방향성 그래프를 DGLGraph 함수를 활용하여 나타낸다. 노드의 index ( ID ) 는 0 부터 연속적으로 시작되어있는 형태로 구성되어 있어야함. 

예를 들어 다음 코드는 방향성 그래프 , 모양은 별모양이며 center node 는 0이다.

```python
g = dgl.graph(([0,0,0,0,0], [1,2,3,4,5]), num_nodes=6)

g = dgl.grah((torch.LongTensor([0,0,0,0,0]), torch.LongTensor([1,2,3,4,5])), num_nodes=6)

g = dgl.graph(([0,0,0,0,0], [1,2,3,4,5]))
```



Note ; DGLGraph 함수에서 방향 그래프를 사용하는것을 추천드립니다. GNN 구조에 message가 한 노드로부터 다른 노드까지 directed하게 send 하는 것이 방향성그래프에서 suitable하게 되었다는것을 관찰하였기 때문에.



## Assigning Node and Edge Features to Graph



많은 그래프 데이터는 노드 그리고 엣지로 이루어져 있다. 'DGLGraph'는 오직 특성을 텐서형태로만 저장하는 것을 원칙으로 함. (with numerical contents).

결과적으로 모든 노드와 엣지는 반드시 같은 shape이여야 함. 

노드와 엣지에 대한 피쳐를 부여하기 위해 'ndata' , 'edata'를 활용하면 된다.

```python
g.ndata['x'] = torch.randn(6,3)
g.edata['a'] = torch.randn(5,4)
g.ndata['y'] = torch.randn(6, 5, 4)
```



### Note ; Data preprocessing at Graph Data

- Categorical attributes (e.g. gender, occpuation), 과 같은 변수들은 one-hot-encoding 을 추천함.

- length string contents (e.g. news article, quote), 과 같은 변수들은 language model을 적용하여 도출된 embedding 값을 추천함.
- image 와 같은 변수들은 vision model로부터 적용하여 도출된 embedding 값을 추천함.

## Querying Graph Structures

```python
print(g.num_nodes())

print(g.num_edges())

print(g.out_degrees(0))

print(g.in_degrees(0))
```



## Graph Transformations

DGL 은 main graph 에서부터 sub graph로 추출하는 기능을 가진 API들이 존재함.

```python
본 그래프의 특성 중 일부인 노드 0 , 1 , 3 으로부터 추출하여 그래프 구성
sg1 = g.subgraph([0, 1, 3])
본 그래프의 특성 중 일부인 엣지 0 , 1 , 3 으로부터 추출하여 그래프 구성
sg2 = g.edge_subgraph([0, 1, 3])
```



## Loading and Saving Graphs

'dgl.save_graphs', 'dgl.load_graphs'를 활용함.

```python
# save
dgl.save_graphs('graph.dgl', g) # single graph
dgl.save_graphs('graphs.dgl', [g, sg1, sg2]) #  multi graph

# load
(g,) , _ = dgl.load_graphs('graph.dgl')
(g, sg1, sg2), _ = dgl.load_graphs('graphs.dgl')
```



# Stochastic Training of GNN for Link Prediction

### Link prediction 

Link prediction 은 노드간에 발생하는 엣지에 대한 존재 확률을 예측하는 것이다. 근접 2개의 노드간의 내적을 통해 계산을 한다. 그때 발생하는 값을 토대로 binary cross entropy loss 를 적용하여 minimize 하는것이 목표다.


$$
\hat{y_{u~v}} = \sigma(h^T_u,h_v)
$$
이 식은 link prediction 에 대한 loss를 generation 한다
$$
L = \sum_{u~v\in D}(Y_{u~v} log(\hat{y}_{u~v})+(1 - {y}_{u~v})log(1-\hat{y}_{u~v}))
$$

### loading dataset

ogb package 로부터 ogbn-arxiv , node classification을 위한 데이터셋을 활용할 예정임. unsupervised link prediction 방법론을 활용하여 embedding (GNN training)을 한 이후에 생성된 node embeddings 를 활용하여 downstream task에 적용할 예정.그 때 우리는 linear classifier 활용하여 최종적으로 arxiv CS paper 의 subject areas 에 대해 예측해보고자 함.

```python
dataset = DGL~~
device = 'cuda' # cpu 사용시에는 cpu , gpu 사용시에는 'cuda'를 꼭 명시해줘야함.

graph, node_labels = dataset[0]
# ogbn-arxiv 는 무방향성 그래프이기에 reverse edges를 추가해준다. 즉 bidirectional graph화 해준다는것을 의미함.
graph = dgl.add_reverse_edges(graph)

node_features = graph.ndata['feat']
node_labels = node_labels[:, 0]
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of classes:', num_classes)

### Key point ###
idx_split = dataset.get_idx_split() ## function이 존재했음.. sklearn train test split을 안해도 되겠읍니다... :)
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']
```



### Defining Neighbor Sampler and Data Loader 

기존 link prediction 예시에서는 full graph 를 mini-batch로 나누어 GNN 을 train 하였다. 허나 본 튜토리얼에서는 인접 노드들간의 representation 을 output화하기 위하여 neighbor sampling 을 하고자 한다. 

이 때 'dgl.dataloading.EdgeDataLoader'을 활용한다.또한 link prediction을 활용하기 위해 'negative sampler'을 특정 해야하며 DGL은 빌트인 negative sampler을 친절하게도 제공한다 'dgl.dataloading.negative_sampler.Uniform'. 본 튜토리얼에서는 5 negative examplers per positive example하고자 한다.

이 때 negative sampling 에 대해 대략적으로 말하자면 주변 노드들을 긍정(positive), 랜덤으로 샘플링 된 노드들을 부정(negative)으로 레이블링 한다는것을 의미함.



```python
negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)

sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])

# argument 잘 기억하기 ! 
train_dataloader = dgl.dataloading.EdgeDataLoader(
	graph, # The graph
    torch.arange(graph.number_of_edges()), # The edges to iterate over
    sampler, # The neighbor sampler
    negative_sampler=negative_sampler, # The negative sampler
    device=device,
    # below argument are same as pytorch dataloader ^^!
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0
)
```



minibatch 는 4가지 특성으로 구성되어 있음.

1. input node의 ID tensor 
2. minibatch 내의 긍정 그래프
3. minibatch 내의 부정 그래프
4. The last element is a list of MFGs storing the computation dependencies  for each GNN layer. The MFGs are used to compute the GNN outputs of the  nodes involved in positive/negative graph.

### Defining Model for Node Representation

이전 task 인 node classification 과 유사하나 차이점은 마지막 output dimension이 기존에는 number of classes 였으나 본 task 인 link prediction에서는 그렇지않음. => (h_feats)



### Defining the Score Predictor for Edges

minibatch 로 부터 node representation을 얻은 이후에 마지막엔 minibatch 내에서 non-existent edge와의 score를 계산한다. 이 과정은 score predictor이라 하며 인접 노드들간의 내적을 통해 나온 값을 활용하여 값이 추출됨.

```python
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between 
            # the source node feature 'h' and destination node feature'h'
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it!
            return g.edata['score'][:, 0]
```



### Evaluating Performance

link prediction에서의 evaluation metric 은 다양하다. 본 튜토리얼에서는 GraphSAGE 논문에 나온 실험을 벤치마킹하여 진행할 예정임. GraphSAGE에서는 training 을 통하여 node embedding을 얻고 이후에 linear classifier 을 모델의 최상위층에 배치하여 embedding 을 평가한다.

```python
# generating node embeddings of each node within the graph
def inference(model, graph, node_features):
    with torch.no_grad():
        nodes = torch.arange(graph.number_of_nodes())
        sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
        train_dataloader = dgl.dataloading.NodeDataLoader(
        	graph, torch.arange(graph.number_of_nodes()), sampler,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            device=device
        )
        result = []
        for input_nodes, output_nodes, mfgs in train_dataloader:
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            result.append(model(mfgs, inputs))
            
       return torch.cat(result)

import sklearn.metrics

def evaluate(emb, label, train_nids, valid_nivds, test_nids):
    classifier = nn.Linear(emb.shape[1], num_classes).to(device) # Linear classification layer
    opt = torch.optim.LBFGS(classifier.parameters())
    
    def compute_loss():
        pred = classifier(emb[train_nids].to(device))
        loss = F.cross_entropy(pred, label[train_nids].to(device))
        return loss
    
    def closure():
        loss = compute_loss()
        opt.zero_grad()
        loss.backward()
        return loss
    
    prev_loss = float('inf')
    for i in range(1000):
        opt.step(closure)
        with torch.no_grad():
            loss = mopute_loss().item()
            if np.abs(loss - prev_loss) < 1e-4:
                print('Converges at iteration', i)
                break
            else:
                prev_loss = loss
                
	with torch.no_grad():
        pred = classifier(emb.to(device)).cpu()
        label = label
        valid_acc = sklearn.metric.accuracy_score(label[valid_nids].numpy(), pred[valid_nids].numpy())
        test_acc = sklearn.metrics.accuracy_score(label[test_nids].numpy(), pred[test_nids].numpy())
    return valid_acc, test_acc
```



### Defining Training Loop

```python
model = Model(node_features.shape[1], 128).to(device)
predictor = DotPredictor().to(device) ## dot-product
opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters())) ## predictior 
```



```python
import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'
for epoch in range(1):
    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat'] ## graph (raw data)

            outputs = model(mfgs, inputs) ## node representation
            pos_score = predictor(pos_graph, outputs) ## 
            neg_score = predictor(neg_graph, outputs) ## 

            score = torch.cat([pos_score, neg_score]) 
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            loss = F.binary_cross_entropy_with_logits(score, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

            if (step + 1) % 500 == 0:
                model.eval()
                emb = inference(model, graph, node_features)
                valid_acc, test_acc = evaluate(emb, node_labels, train_nids, valid_nids, test_nids)
                print('Epoch {} Validation Accuracy {} Test Accuracy {}'.format(epoch, valid_acc, test_acc))
                if best_accuracy < valid_acc:
                    best_accuracy = valid_acc
                    torch.save(model.state_dict(), best_model_path)
                model.train()

                # Note that this tutorial do not train the whole model to the end.
                break
```



#### Message Passing Review

$$
m^{(l)}_{u\rightarrow v}=M^{l}(h^{(l-1)}_v,h^{(l-1)}_u,e^{(l-1)}_{u\rightarrow v})
$$

$$
m^{(l)}_v = \sum_{u\in N(v)}m^{(l)}_{u\rightarrow v}
$$

$$
h^{(l)}_v=U^{(l)}(h^{(l-1)}_v,m^{(l)}_v)
$$

- M^{(l)} is the message function

- \sum is the reduce function
- \U^{(l)} is the update function
- Note that \sum here can represent any function and is not necessarily a summation.

** Essentially, the l 번째 레이어의 노드는 l-1 레이어의 노드 output 에 의존한다. , 더불어 l-1 번째의 연결된 주변 이웃에도 의존하게 된다. 이 때 l-1 번째 노드는 l-2번째에 연결된 노드 , 그리고 그 노드와 연결된 주변이웃에 의존한다. 앞 선 과정들을 토대로 update됨.



#### Neighbor Sampling Overview

이전 message passing tutorial 에서 봤듯이 small node를 update하기 위해서는 많은 노드에 의존하기에 이는 기하급수적으로 computing cost가 늘어남. 이 현상을 해결하고자 Neighbor sampling을 적용한다. [scale-free](https://en.wikipedia.org/wiki/Scale-free_network) 라는 이론의 가정을 통해 neighbor sampling 으로 해결해보고자 함.

![image-20210601132313708](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210601132313708.png)

from ; https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sw4r&logNo=221273784494



보다 자세히 알아보기 위해서 Random 네트워크와 Scale-Free 네트워크를 비교하면서 알아보자. 랜덤 네트워크의 경우, n이 무한대로 가면 즉, 노드가 무한개인 네트워크를 고려하면, 포아송 분포를 따르게 된다. 아래 그래프에서 보이듯이, 특정한 degree k 값에서 확률이 가장 높은 즉, 가장 지배적인 degree가 네트워크 내에서 존재하는 것이다.

그래서 아래 네트워크 구조를 보면, 랜덤으로 한 노드를 정하면 거의 2의 degree가 나오는 것을 확인할 수 있다. 반면, Scale Free 네트워크의 경우에는 Power Law 분포를 따르게 되면, 분포의 가장 큰 특징으로는 어느 정도의 Hub가 존재하게 되는데, 이것들은 Degree가 높기 때문에 상당히 많은 노드들과 연결을 가지고 있게 된다. 반면, 대부분의 다른 노드들은 상당히 낮은 Degree를 보유하는 형태가 된다.

![image-20210601132340132](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210601132340132.png)



Neighbor sampling 는 앞선 현상을 이웃들 중 subset을 선택하여 aggregation을 실행한다. 



![](https://data.dgl.ai/tutorial/img/bipartite.gif)



# Message passing

$$
h^{k}_{N(v)} \leftarrow Average{h^{(k-1)}_u,\forall u\in N(v)}
$$

$$
h^k_v <- ReLU(W^k \cdot CONCAT(h^{k-1}_v,h^k_{N(v)}))
$$

DGL 빌트인 function인 'dgl.nn.pytorch.SAGEConv' 사용해서 앞선 aggregation을 간단하게 사용할 수 있음.



Here is how you can implement GraphSAGE convolution in DGL by your own.



```python
class SAGEConv(nn.Module):
    '''Graph convolution module used by the GraphSAGE model.
        
    Parameters
    ----
    in_feat ; int ( input feature size )
    out_feat ; int ( output feature size )
    
    '''
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2 , out_feat)
        
    def forward(self, g, h):
        '''Forward computation
        
        Parameters
        ----
        g ; Graph , The input graph
        h ; Tensor , The input node feature.
                
        '''
        with g.local_scope():
            g.ndata['h'] = h # feature
            # update_all is a message passing API
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N')) # 'fn' means  dgl function 
            h_N = g.ndata['h_N'] # updated feature
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

- 'g.update_all' function 에 대해 부연설명, 주변 이웃의 feature을 gather and average 의 역할을 담당함. 크게 3가지 concept로 활용함.
  - Message function ; "'fn.copy_u('h', 'm')" function  - node feature을 copy하고  messages sent to neighbors 에게 전송해주는 함수
  - Reduce function ; "fn.mean('m', 'h_N')" function -  받은 node feature(h -> m ) 'm'를 토대로  새로운 node feature 'h_N' 에 save해줌.
  - 'update_all' 앞선 function 들 (Message function, Reduce function)에 대해 trigger역할을 함.

Module(layer) 구성이 끝나고 Modeling 을 해준다 .



```python
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)
        
        # g is graph , in_feat is node-feature
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```



```python
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

def train(g, model):
    optimizer = torch.optim.ADam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    # mask 는 앞 서 설정한 get_idx_split으로부터 ...?
    '''
    
idx_split = dataset.get_idx_split() ## function이 존재했음.. sklearn train test split을 안해도 되겠읍니다... :)
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']
    '''
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(200):
        # Forward
        logits = model(g, features)
        
        # Compute prediction
        pred = logits.argmax(1)
        
        # Compute loss
        # note that we should only compute the losses of the nodes in the training set, 
        # i.e. with train_mask 1
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        # Compute accuracy on train/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach()) ## gpu -> cpu , tensor -> numpy for printing
        
        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

            
# input , hidden , num_classes
model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```



## Customization



** Note that 'edata' member 는 엣지 특성을 message passing 안에 접목을 해줌.

```python
class WeightedSAGEConv(nn.Module):
    '''Graph convolution module used by the GraphSAGE model with edge weights.
    
    Parameters
    ----
    in_feat ; int input feature size
    out_feat ; int output feature size
	'''
    
    def __init__(Self, in_feat, out_feat):
        super(WeightedSAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output
        self.linear = nn.Linear(in_feat * 2, out_feat)
        
    def forward(self, g, h, w):
        '''Forward computation
        
        Parameters
        ----
        g ; Graph -> graph
        h ; Tensor -> node feature
        w ; Tensor -> The edge weight
        
        '''
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(message_func = fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m','h_N')) ## edge weight 인 'w'이 포함되어있음.
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

Because the graph in this dataset does not have edge weights, we manually assign all edge weights to one in the 'forward()' function of the model. 

-> 임의적으로 edge weight를 생성해서 추가해주는게 가능함. (!!!!????)

```python
class Model(nn.Module):
def __init__(self, in_feats, h_feats, num_classes):
    super(Model, self).__init__()
    self.conv1 = WeightedSAGEConv(in_feats, h_feats)
    self.conv2 = WeightedSAGEConv(h_feats, num_classes)
    
def forward(self, g, in_feat):
    h = self.conv1(g, in_feat, torch.ones(g.num_edges()).to(g.device))
    h = F.relu(h)
    h = self.conv2(g, h, torch.ones(g.num_edges()).to(g.device))
    return h

model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```

#### Even more customization

DGL allows user-defined message and reduce function for the maximal expressiveness. 

user-defined message function -> fn.u_mul_e('h', 'w', 'm').

```python
def u_mul_e_udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}
```

edges 는 3가지 constitute 가 존재함 src, data 그리고 dst  각각 시작노드피쳐 , 엣지피쳐 그리고 도착노드피쳐를 의미한다. 

You can also write your own reduce function. For example, the following is equivalent to the builtin ``fn.sum('m', 'h')`` function that sums up the incoming messages:

```python
def sum_udf(nodes):
    return {'h': nodes.mailbox['m'].sum(1)}
```

In short, DGL will group the nodes by their in-degrees, and for each group DGL stacks the incoming messages along the second dimension. You can then perform a reduction along the second dimension to aggregate messages.

# Stochastic Training of GNN with Multiple GPUs







# Link Predict

This tutorial은 어떻게 GNN이 link prediction task를 위해 적용하는지에 대해 배워보고자 함. i.e. 그래프 내의 임의의 2개 node간에 edge가 존재할지 안할지 예측

## Overview of Link Prediction with GNN

social recommendation, item recommendation, knowledge graph completion 와 같은 application 은 대다수 link prediction으로 구성되어 있음. 두개의 특정 노드사이간에 edge가 형성될지 안될지에 대한 예측. 본 튜토리얼은 citing 할지 cited 인제 애대 2개의 페이퍼간 의 관계에 대해 예측해보고자 함.

This tutorial formulates the link prediction problem as a binary classification problem as follows;

- 그래프 내의 실제 존재하고 있는 edges는 positive example로 간주함.
- node 쌍에서 edge가 존재하고 있지 않는 것을 sampling 함. 이걸 negative example.
- 앞서 나눈 positive example과 negative example을 training set 과 test set으로 나누어줌.
- 최종적으로 모델 성능을 평가하기 위해 binary classification 성능지표인 AUC를 활용하여 평가함. 

In some domains such as large-scale recommender systems or information retrieval, you may favor metrics that emphasize good performance of top-K predictions.

## Loading graph and features



## Prepare training and testing sets

이 튜토리얼은 test set에 있는 10% 만을 랜덤으로 추출하고 남은 것을 training set으로 가정. 이 때 학습시킬 negative sample 도 필요한데 len(positive samples) == len(negative samples) 가 되어야함.

```python
# split edge set for training andd testing
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)  ## permutation 

'''numpy.random.permutation(x)
Randomly permute a sequence, or return a permuted range.

Parameters
x ; int or array_like 

Returns
out ; ndarray
'''
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size

test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u , train_pos_v = u[eids[test_size:], v[eids[test_size:]]]


## Negative Sampling

### 자주 헷갈리는 부분 아래 프로세스는 adjacency matrix 를 만든다.
### 만든 matrix 에서 1을 빼면 neg된 adjacency matrix 로 만들어짐. 
### neg_u, neg_v 여기에서 u는 source , v는 destination을 의미하는거라 생각됨.

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
'''scipy.sparse.coo_matrix
A sparse matrix in COOrdinate format.

Also known as the 'ijv' or 'triplet' format.

Example
coo.matrix((3,4), dtype=np.int8).toarray()
array([[0,0,0,0],
		[0,0,0,0],
		[0,0,0,0]], dtype=int8)
'''
## adj neg를 만드는 과정 
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
## adj.todense()는 adjacency를 의미 ! (위 코드에서 adj 생성시 u.numpy(), v.numpy() 형식으로 만들어서 matrix화 했음!) 
## 즉, adj_neg 는 negative 
neg_u , neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)

test_neg_u , test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u , train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

```

** 윗 형식과 같음. 

![image-20210603180021370](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210603180021370.png)

** 각각의 element 에 대한 

![image-20210603181553695](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210603181553695.png)

## Define a GraphSAGE model

이전에 layer modeling 에 대해 작성하였으므로 중복되어 생략하겠음.

결론적으로 모델은 edge score를 계산하고자 하는데 나타내는 function은 다양함 ( e.g. an MLp or a dot product)
$$
\hat y_{u~v} =f(h_u,h_v)
$$


## Positive graph, negative graph, and 'apply_edges'

이전 tutorial 인 node classification 과 다르게 link prediction은 한 쌍의 노드가 필요함!

DGL 은 앞선 한쌍의 노드를 각각의 graph에 존재하다고 가정함. 그리하여 link prediction 에서는 positive graph ( pos 로만 구성된 것) , negative graph ( neg 로만 구성된 것으로 나누어 진행). 두 그래프는 original graph으로부터 subset 된거임. 여기에 대해 언급한 이유는 node feature을 적용할때 좀 더 효용적으로 진행하고자 함. 



```python
# constructing the pos graph , the neg graph
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
```

The benefit of treating the pairs of nodes as a graph is that you can use the ``DGLGraph.apply_edges`` method, which conveniently computes new edge features based on the incident nodes’ features and the original edge features (if applicable).

DGL provides a set of optimized builtin functions to compute new edge features based on the original node/edge features. For example, ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’ representations for each edge.

```python
import dgl.function as fn

calss DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by dot-product
            # between the source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edat['score'][:, 0]
```



```python
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
        
    def apply_edges(self, edges):
        '''apply_edges
        Computes a scalar score for each edge of the given graph.
        
        Parameters
        ----
        edges ; ''src'', ''dst'' and ''data'', 각각의 형태들은 사전형태로 저장되어짐.
        
        returns
        ----
        dict ; A dictionary of new edge features.
        '''
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
    
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
```

** 되도록 dgl builtin function을 쓰는것을 권고드립니다 . speed and memory 측면에서 class에 맞게 최적화되어있기 때문에 그렇습니다.

## Training loop

node representation computation 과 edge score computation을 정의 한 이후에 모델 그리고 loss function 그리고 metric 을 지정해야합니다. 

loss function 으로는 간단하게 binary cross entropy loss를 이용할 것임.
$$
L = - \sum_{u~v\in D}(y_{u~v}log(\hat y_{u~v})+(1-y_{u~v})log(1-\hat y_{u~v})))
$$
evaluation metric 으로는 AUC 를 사용할것임.



```python
model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
# MLPPredictor 를 DotPredictor로 바꿀수도 있음.
# pred = MLPPredictor(16)
pred = DotPredictor()

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_corss_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
    	[torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(albels, scores)
```



```python
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)

all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if e % 5 == 0:
        print(f'In epoch {e}, loss: {loss})
              
from sklearn.metrics import roc_auc_score
with torch.no_grad():
     pos_score = pred(test_pos_g, h)
     neg_score = pred(test_neg_g, h)
     print('AUC', compute_auc(pos_score, neg_score))
```





# Load_data



## Make your own dataset

DGLDataset Object Overview

graph dataset은 dgl.data.DGLDataset 클래스로부터 상속받습니다 그리고 다음과 같은 method(argument)를 갖습니다.

- ''getitem(self,i)'' ; i번째 데이터셋를 가져와 single DGL graph 에 포함시키고 때때로 label 도함.
- ''len(self)'' ; datasets 의 갯수
- ''process(self)'' ; disk 로 부터 raw data를 불러들이고 process함..

```python
import dgl
from dgl.data import DGLDataset
import torch
import os

class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')
        
    def process(self):
        nodes_data = pd.read_csv('./members.csv')
        edges_data = pd.read_csv('./interactions.csv')
        node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features
        
        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1

dataset = KarateClubDataset()
graph = dataset[0]

print(graph)
```





