# DGL-winter-school



본 리뷰는 [dgl-winter-school](https://github.com/dglai/dgl_winter_school) 의 게시물을 참조하였습니다.



# Semi-supervised node classification using GNN



## Semi-supervised node classification ; problem statement

- 주어진 그래프 구조중 모든 노드의 특성 그리고 노드의 서브셋 라벨(sampling)을 토대로 남은 노드의 라벨을 예측하는 작업.



## Classification in citation networks

데이터셋  Cora dataset

- 2708 개의 특정 논문을 7가지의 클래스로 구성함.
- 10556개의 엣지로 이루어져 있음.
- 각각 논문들의 특성은 abstract의 word들을 vector화 함.

## Deinfe a GCN model 

### Message passing interpretation of GCN

$$
m_v^{(l)}=\sum_{w\in N(v)}M^{(l)}(h^{(l-1)}_v,h^{(l-1)}_w,e_{vw}) \leftarrow (1)
$$


$$
h^{(l)}_v=U^{l}(h^{(l-1)}_v,m^{l}_v) \leftarrow (2)
$$
where $\sum$ 은 Reduce/Aggregate 를 의미함.

$M^{(l)}~ $는 Message를 의미함.

equation (2) 에서 $U^{(l)}$ 은 Update 를 의미함.

**GCN update functions**
$$
M^{(l)}_{vw}=\frac{h_w^{(l-1)}}{d_v+1}
$$

$$
m^{(l)}_v=\sum_{w\in N(v)\cup{{\{v\}}}}M^{(l)}_{vw}
$$

$$
h^{(l)}_v = \phi(m^{l}_vW^{l})
$$

$M_vw^{(l)} 에 대한 수식 정의 이해가 어려움. d_v 가 무엇인지에 대한 갈피를 못잡겠음. 아마 Message 즉 그 노드내의 feature이지 않을까 추측중.

$m_v^{(l)}$ 은 앞선 M_vw에 대한 정보를 모두 합쳐놓음. (reduct function)

$h_v^{(l)}$ message와 learnable parameter 인 W를 업데이트 해줌.



### GCN layer in DGL



base layer 2개 

```
Epoch 00000 | Loss 1.9458 | Accuracy 0.0780 | 
Epoch 00020 | Loss 1.5064 | Accuracy 0.4700 | 
Epoch 00040 | Loss 0.6011 | Accuracy 0.6660 | 
Epoch 00060 | Loss 0.1329 | Accuracy 0.7180 | 
Epoch 00080 | Loss 0.0359 | Accuracy 0.7260 | 
Epoch 00100 | Loss 0.0157 | Accuracy 0.7220 | 
Epoch 00120 | Loss 0.0091 | Accuracy 0.7220 | 
Epoch 00140 | Loss 0.0062 | Accuracy 0.7200 | 
Epoch 00160 | Loss 0.0045 | Accuracy 0.7200 | 
Epoch 00180 | Loss 0.0035 | Accuracy 0.7220 | 
```

layer + 1 즉 hop 3개 , 주변 이웃 3번을 건넌 사람들로부터 정보를 수합했을 때.

```
Epoch 00000 | Loss 1.9455 | Accuracy 0.1820 | 
Epoch 00020 | Loss 1.5343 | Accuracy 0.6340 | 
Epoch 00040 | Loss 0.8846 | Accuracy 0.7280 | 
Epoch 00060 | Loss 0.3948 | Accuracy 0.7700 | 
Epoch 00080 | Loss 0.1732 | Accuracy 0.7780 | 
Epoch 00100 | Loss 0.0885 | Accuracy 0.7780 | 
Epoch 00120 | Loss 0.0531 | Accuracy 0.7800 | 
Epoch 00140 | Loss 0.0358 | Accuracy 0.7840 | 
Epoch 00160 | Loss 0.0260 | Accuracy 0.7800 | 
Epoch 00180 | Loss 0.0199 | Accuracy 0.7780 | 
```



[optimizer](https://sanghyu.tistory.com/113) 에 learning schedular 기능을 활용하였을때 

```python
# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
# optimizer
optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=0.01)
## learning schedular
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                       lr_lambda=lambda epoch : 0.95 ** epoch,
                                       last_epoch=-1,
                                       verbose=False)
loss_fcn = torch.nn.CrossEntropyLoss()
# ----------- 4. training -------------------------------- #
n_epochs=200
for epoch in range(n_epochs):
        model.train()

        # forward
        logits = model(g,features)
        # loss
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        acc = evaluate(g,model, features, labels, val_mask)
        if epoch%20==0:
            print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | ". format(epoch, loss.item(),
                                             acc))
print()

```



learning schedular에 따른 뚜렷한 성능향상 저하에 대한 것은 확인치 못함.



# Link Prediction using GNN



## Link prediction ; Problem statement



```python
data = CoraGraphDataset()
g = data[0]
features = g.ndata['feat']
labels = g.ndata['label']

in_feats = features.shape[1]
n_classes = data.num_classes
n_edges = data.graph.number_of_edges()
```



- 그래프 구조와 노드 특성이 주어졌을때 그래프 내의 두 노드가 연결되는지 안되는지에 대해 예측

  

## Prepare training and testing sets.

**Permutation** -> shuffle 의 기능을 함.

![image-20210604182642083](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210604182642083.png)



## Negative links

- 긍정 edge는 GNN 이 일정 node 와 연결되었는가를 촉진시키는 역할을 함.
- 부정 edge는 GNN으로부터 특정 node와 연결되어있지 않는것을 훈련시키는 역할을 함.
- 부정 edge는 그래프 내의 연결되어 있지 않은 엣지로부터 샘플링 됨.
- 현재 negative sampling을 어떻게 하는게 좋을것인지에 대해 연구쪽에서는 활발한 연구가 이루어 지고 있음.
- 우리는 missing edges를 enumerate 하여 500개는 testing , 500개는 training 해보고자 함.

```python
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(adj.shape[0])
neg_u , neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), 2000)
test_neg_u, test_neg_v = neg_u[neg_eids[:500]], neg_v[neg_eids[:500]]
train_neg_u, train_neg_v = neg_u[neg_eids[500:]], neg_v[neg_eids[500:]]
```



```python
train_u = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
train_v = torch.cat([torch.as_tensor(train_pos_v), torch.as_tensor(train_neg_v)])
train_label = torch.cat([torch.zeros(len(train_pos_u)), torch.zeros(len(train_neg_u))])

test_u = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
test_v = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
test_label = torch.cat([torch.zeros(len(test_pos_u)), torch.zeros(len(test_neg_u))])
```



## Define a GCN model







## Link prediction loss function

$$
\hat y_{u~v} = \sigma(h^T_uh_v)
$$

$$
L = - \sum_{u~v\in D}(y_{u~v}log(\hat y_{u~v})+(1-y_{u~v})log(1-\hat y_{u~v})))
$$

- 모델은 각각의 노드 임베딩 내적곱을 통하여 엣지 확률을 예측한다
- binary cross entropy 는 타겟 y 에 대해 0, 1 인지에 대해 positive , negative 로 학습하게끔 만들어주는 loss 이기에 link prediction 에 적합하다 생각하여 사용함.





/divide

# ☆☆☆ Heterogeneous graphs in DGL ☆☆☆

이번 튜토리얼에서 배울 것들.

- 외부 파일로부터 어떻게 DGL heterogeneous graph 를 생성하는지에 대한 것.
- DGL heterogeneous graph 의 특성에 대해 어떻게 접근하고 수정하는지에 대한 것.
- [DRKG](https://github.com/gnn4dr/DRKG) drug discovery , knowledge graph 





## Heterogeneous vs. Homogeneous

- 이종 vs. 동종 으로 해석되는데 간단히 설명해보면 이종그래프는 노드 , 엣지가 각각 다양한 타입으로 이루어진 데이터를 의미함. 동종그래프는 노드 , 엣지가 각각 동일 타입으로 이루어져 있음 . 다음 figure를 보면 이해가 쉬울듯 하다. (다양성 측면에서 더욱 detail하게 aggregation 이 될것임. ) knowledge graph 도 heterogeneous 카테고리에 속함.

example ) 

![Knowing Your Neighbours: Machine Learning on Graphs | by Pantelis Elinas |  stellargraph | Medium](https://miro.medium.com/max/1754/1*DfoOgPPusJAUm_kSN8O_mA.png)

figure from ; [source](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896)



## Loading the drug repurposing knowledge graph in dgl



'  knowledge graph transformation code '

* triplet 으로 변환해주는게 주 목표임 .

```python
def create_drkg_edge_lists():
    download_and_extract()
    path = "../data/"
    filename = "drkg.tsv"
    drkg_file = os.path.join(path, filename)
    df = pd.read_csv(drkg_file, sep ="\t", header=None)
    triplets = df.values.tolist()
    entity_dictionary = {}
    def insert_entry(entry, ent_type, dic):
        if ent_type not in dic:
            dic[ent_type] = {}
        ent_n_id = len(dic[ent_type])
        if entry not in dic[ent_type]:
             dic[ent_type][entry] = ent_n_id
        return dic

    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]
        insert_entry(src,src_type,entity_dictionary)
        insert_entry(dest,dest_type,entity_dictionary)

    edge_dictionary={}
    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]

        src_int_id = entity_dictionary[src_type][src]
        dest_int_id = entity_dictionary[dest_type][dest]

        pair = (src_int_id,dest_int_id)
        etype = (src_type,triple[1],dest_type)
        if etype in edge_dictionary:
            edge_dictionary[etype] += [pair]
        else:
            edge_dictionary[etype] = [pair]
    return edge_dictionary
```

그렇게 변환된 edge list 를 토대로 생성된 edge_list_dictionary

![image-20210604191053835](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210604191053835.png)

-  src , ent_type , dest 의 형식으로 구성되어 있음. 



### Create a DGL graph from edge lists

- 주어진 edge list로 graph를 생성해낸다.
- 그래프의 스키마를 통해 Metagraph (graph 의 메타형식) 를 정의할 수 있음.

```python
graph = dgl.heterograph(edge_list_dictionary);
print(graph)
```

위 코드의 결과는 다음과 같음.

![image-20210604191411089](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210604191411089.png)



### ID of heterogeneous graph nodes spaces

- 각자 다른 타입의 노드와 엣지는 각자 독립적인 ID 공간과 feature 공간을 가지고 있음. 
- Drug 그리고 Protein node IDs 는 0부터 시작하며 각각 다른 feature를 가지고 있음.

Question ? knowledge graph 의 feature 은 어떻게 적용하는걸까 ? 단순 edge type 으로만 적용이 가능한것인가?

### Print the statistics of the created graph

- node type

![image-20210604191842070](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210604191842070.png)

- edge type

![image-20210604191854064](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210604191854064.png)

### Assigning node features

```python
graph.nodes['Compound'].data['hv'] = torch.ones(graph.number_of_nodes('Compound'), 1)
graph.edges['DRUGBANK::treats::Compound:Disease'].data['he'] = torch.zeros(graph.number_of_edges('DRUGBANK::treats::Compound:Disease'),1)
# node feature
print(graph.nodes['Compound'].data['hv'])
# edge feature
print(graph.edges['DRUGBANK::treats::Compound:Disease'].data['he'])
```



# Semi-supervised node classification using Heterogeneous Graph Neural Networks

이번 튜토리얼에서 배울 것.

- Building the Relational graph neural network model by [RGCN](https://arxiv.org/abs/1703.06103)

- 모델을 훈련시키고 결과에 대한 이해

## 문제정의

- 특정 타입의 노드에서 서브셋을 토대로 주어진 그래프 구조, 노드 특성 그리고 노드 레이블 을 통해 남은 nodes 의 labeled type을 예측하는 것.

```python
dataset = AIFBDataset()
g = dataset[0]

category = dataset.predict_category
num_classes = dataset.num_classes

# obatin the training testing splits stored as graph node attributes
train_mask = g.nodes[category].data.pop('train_mask')
test_mask = g.nodes[category].data.pop('test_mask')
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
labels = g.nodes[category].data.pop('labels')

# split dataset into train, validate, test
val_idx = train_idx[:len(train_idx) // 5]
train_idx = train_idx[len(train_idx) // 5:]

# check cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
g = g.to(device)
labels = labels.to(device)
train_idx = train_idx.to(device)
test_idx = test_idx.to(device)
```



## Heterogeneous models

- Heterogeneous graph는 다양한 엣지 타입을 가지고 있음.
- 다른 엣지타입으로부터 메세지를 받는다.
- 모델은 각각의 edge-type 으로부터 받는 message를 어떻게 종합할 것인지에 대해 정의할 수 있음.



### Relational GCN model

- RGCN 은 각각의 edge relation type마다의 메세지를 모두 더함 .

$$
h^{(l+1)}_i = \sigma(\sum_r\sum_{j\in N^r_{(i)}}\frac{1}{c_{i,r}}h^{l}_jW^{l}_r)
$$

- 위의 equation에 따라 HeteroRGCNLayer 이 적용된다.

- dictionary 노드 타입과 노드 피쳐를 input 하여 다른 dictionary 노드 타입과 노드 피쳐를 가져옴.
- For a graph with R relations it uses
  - R message passing functions
  - R aggregation functions
  - A single function to aggregate the messages across relations.



# **☆☆☆☆☆  레이어 구성이해하기. **

```python
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                # Specify per-relation message passing functions: (message_func, reduce_func).
                # Note that the results are saved to the same destination feature 'h', which
                # hints the type wise reducer for aggregation.
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.dstnodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.dstnodes[ntype].data}
```



### Deffnie a HeteroGraphConv model

- HeteroGraphConv 는 DGL nn 모듈에 내장되어 있음.
  - f_r(\cdot, \cdot): DGL NN 모듈은 각각의 relation을 정의해줘야함.
  - A DGL NN 모듈은 message passing 그리고 aggregation function과 일치함.
  - G(\cdot) ; reduction function 은 다른 relation 으로부터 같은 노드 타입을 합쳐주는 역할을 함. 
  - g_r ; Graph 의 각 relation r 을 의미함.

$$
h^{(l+1)}_x = G_{r\in R, r_{dst}=x}(f_r(g_r,h^l_{r_{src}},h^l_{r_{dst}}))
$$



```python
# 모델 생성
import dgl.nn as dglnn

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        
    def forward(self, graph, inputs):
        # 여기에서 인풋은 모두 node 의 피쳐를 의미함.
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v, in h.items()}
        h = self.conv2(graph, h)
        return h
```



### Flexbility of HeteroGraphConv

- graph convolution 을 각 edge type 에 적용할 수 있음.
- 모든 node type의 final result 처럼 각각의 edge type을 message aggregation처럼 활용하여 더할 수 있음.
- GraphConv 를 GraphAtt 로 대체함으로써 다른 모델링을 할  수 있음.

### Node embedding layer for heterogeneous graph

- AIFB 는 node feature가 없기때문에 우리는 learnable parameter로 embedding을 활용할 것임.
- embedding은 훈련동안 업데이트 될것임.

# ☆☆☆ node feature 가 없을 떄  !!!

```python
class NodeEmbed(nn.Module):
    def __init__(self, num_nodes, embed_size, device):
        super(NodeEmbed, self).__init__()
        self.embed_size = embed_size
        self.node_embeds = nn.ModuleDict()
        self.device=device
        self.num_nodes=num_nodes
        for ntype in num_nodes:
            node_embed = torch.nn.Embedding(num_nodes[ntype], self.embed_size)
            nn.init.uniform_(node_embed.weight, -1.0, 1.0)
            self.node_embeds[str(ntype)] = node_embed
            
    def forward(self):
        embeds = {}
        num_nodes=self.num_nodes
        for ntype in num_nodes:
            embeds[ntype] = self.node_embeds[ntype](torch.tensor(list(range(num_nodes[ntype]))).to(self.device))
        return embeds
```



```python
num_nodes = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}

h_hidden=16
embed = NodeEmbed(num_nodes, h_hidden,device).to(device)
model = RGCN(h_hidden, h_hidden, num_classes,g.etypes).to(device)

```



```python
# ----------- 3. set up optimizer -------------- #

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), embed.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(50):
    # forward
    embeds = embed()
    logits= model(g,embeds)[category]
    
    # compute loss
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())
    
    if e % 5 == 0:
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f}".
              format(e, train_acc, loss.item(), val_acc, val_loss.item()))
```

```python
# ----------- 5. check results ------------------------ #
model.eval()
embed.eval()
embeds = embed()
logits= model.forward(g,embeds)[category]
test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
print()
```



















