Stanford Computer Forum - Graph Learning Workshop



## Section 1 



21 / 09 / 17 / Friday 01:00 am ~ 

pain point # 1 

- **Data scientists still hand encode features to solve prediction problems.**



pain point # 2

- **Data is often incomplete**



'How can we develop neural networks that are much more broadly application?'

-> Graphs are the new frontier of deep learning

-> Graphs are connected



This workshop subject ; Representation of graphs

Applications list

- Graphics and Vision
- Fraud and intrusion detection
- Financial networks
- Knowledge Graphs and Reasoning
- NLP
- biomedicine

Tools 

PyG 2.0 , GraphGym , OGB



goal ; Representation Learning

**Map nodes to d-dimensional embeddings such that similar nodes in the network are embedded close together**



Problem setup

- V
- A
- X



A Naive Approach

- Join adjacency matrix and features
- Feed them into a deep neural net

Issues with this idea

- O(|V|) parameters
- Not applicable to graphs of different size
- sensitive to graph ordering

-> solution & motivation ; Convolutional networks



- Real-World graphs
  - There is no fixed notion of locality or sliding window

Networks as computation graphs 

Key idea ; Network is a computation graph

**Learn how to propagate information across the network**

Each node defines a computation graph

- Each edge in this graph is a transformation / aggregation function





**Inductive Capability**

The same aggregation parameters are shared for all nodes;



**Key Benefits**

- No manual feature engineering needed
- End-to-end learning results in optimal features.
- Any graph machine learning task
- Scalable to billion node graphs

- GNNs adapt to the **shape** of data
  - Other Deep learning architecture assume fixed input ( matrix, sequence ) -> GNN makes not such assumptions



## Section 2



- scaling up model via Pytorch Lightning
- Explainability via Captum



Design Principles

![img](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/architecture.svg?sanitize=true)





Message Passing Graph Neural Networks
$$
h^{(l+1)}_i = Update_{\theta}(h^{(l)}_i \bigoplus   MESSAGE_{\theta}(h^{(l)}_j, h^{(l)}_i, e_{j,i}))
$$


- PyG supports mini-batching on many small graphs

- PyG supports mini-batching on single giant graphs



TorchScript

- Convert pure Python GNN model to an optimized and standalone program



PyTorch Lightning

- choose a GNN model
- Setup a trainer
- Call trainer.fit



Captum

- Explain GNN predictions out-of-the-box



Heterogeneous Graph Support

- Data Storage
  - Holds information about different node and edge types in individual containers
  - Edge types are described by a triplet of source node, relation and destination node type
  - Transformations enhance the graph for message passing, e.g., by adding reverse edges

- Heterogeneous Graph Neural Networks

A homogeneous GNN can be converted to a heterogeneous one by learning distinct parameters for each individual edge type;
$$
h^{(l+1)}_i = \sum_{r\in R}GNN^{(r)}_\theta(h^{(l)}_i, \{{h^{(l)}_j}:j \in N^{(r)}(i)\})
$$


![../_images/to_hetero.svg](https://pytorch-geometric.readthedocs.io/en/latest/_images/to_hetero.svg)



Rapid growth in the number of parameters w.r.t. number of relations may lead to overfitting on rare relations

Basis-decomposition for regularization

$$
h^{(l+1)}_i = \sum_{r\in R}GNN^{(r)}_\theta(h^{(l)}_i, \{{a^{(l)}_{r,b} \times h^{(l)}_j}:j \in N^{(r)}(i)\})
$$

where a^{(l)}_{r,b} means that the **Relational-depend trainable coefficients**



PyG can automatically convert homogeneous GNNs to heterogeneous ones

1. Duplicates message passing modules for each edge type
2. Transforms the underlying computation graph so that message are exchanged along different edge type
3. Use lazy initialization (-1) to handle different input feature dimensionalities



Heterogeneous Graph Samplers

Scaling up heterogeneous GNNs to large-scale graphs with ease via relational neighhbor sampling 

**Only requires a few lines of code change!**



GraphGym

**Design Space Exploration with GraphGym Which GNN is the best for your given task?**



## Section 3
