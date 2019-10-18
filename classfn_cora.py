from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv
import ipdb
import logging
import random
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import SAGEConv, GATConv

class Net_adv(torch.nn.Module):
    def __init__(self):
        super(Net_adv, self).__init__()
        base_dim = 256
        self.conv1 = GATConv(object_dim, base_dim, heads=4)
        self.lin1 = torch.nn.Linear(object_dim, 4 * base_dim)
        self.conv2 = GATConv(4 * base_dim, base_dim, heads=4)
        self.lin2 = torch.nn.Linear(4 * base_dim, 4 * base_dim)
        self.conv3 = GATConv(
            4 * 256, dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * base_dim, dataset.num_classes)

    def forward(self, data, node_feats):
        x_old, edge_index = data.x, data.edge_index
        x = node_feats
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return F.log_softmax(x, dim=1)
        
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        self.conv1 =  DenseGCNConv(object_dim, 32) 
        self.conv12 =  DenseGCNConv(32, 64)
        self.conv13 = GCNConv(64, 128)
        self.conv14 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, dataset.num_classes)

    def forward(self, data, node_feats):
        x_old, edge_index = data.x, data.edge_index
        x = node_feats

        # ipdb.set_trace()

        adj = to_dense_adj(edge_index)

        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.conv12(x, adj)
        x = F.relu(x)

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv12(x, edge_index)
        # x = F.relu(x)
        x = self.conv13(x.squeeze(), edge_index)
        x = F.relu(x)
        x = self.conv14(x, edge_index)
        x = F.relu(x)

        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset_GT = Planetoid(root='/tmp/Cora', name='Cora')

dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
dataset_GT = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')

ipdb.set_trace()

num_classes = 7
give_only_feats = True  
give_feats = True
corrupt = True
if(corrupt):
    p = 0.3        
    num_incorrect = 0
    for i in range(dataset[0].y.shape[0]):
        if(random.uniform(0, 1)<p):
            r = list(range(0, dataset[0].y[i])) + list(range(dataset[0].y[i]+1, num_classes))
            dataset[0].y[i] = random.choice(r)
            num_incorrect += 1

n_objects  = dataset[0].y.shape[0] # number of nodes
y_onehot = torch.zeros((n_objects, num_classes))
y_onehot.scatter_(1, dataset[0].y.unsqueeze(1), 1)                                       

y_onehot_GT = torch.zeros((n_objects, num_classes))
y_onehot_GT.scatter_(1, dataset_GT[0].y.unsqueeze(1), 1)   

# augment x
objects = dataset[0].x  # node features
if(give_feats):
    if(give_only_feats):
        object_dim = num_classes # features
        objects = y_onehot      
    else:
        object_dim = dataset[0].x.shape[1] + num_classes # features
        objects = torch.cat((objects, y_onehot), dim = 1)
else:
    object_dim = dataset[0].x.shape[1] # features

objects = objects.to('cuda:0')
data = dataset[0].to('cuda:0')
data_GT = dataset_GT[0].to('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_adv().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(1500):
    optimizer.zero_grad()
    out = model(data, objects)
    loss = F.nll_loss(out[data_GT.train_mask], data_GT.y[data_GT.train_mask])
    # loss = F.nll_loss(out[data_GT.test_mask], data_GT.y[data_GT.test_mask])

    loss.backward()
    print(loss)
    optimizer.step()

    _, pred = out.max(dim=1)
    # train_correct = float (pred[data_GT.test_mask].eq(data_GT.y[data_GT.test_mask]).sum().item())
    # train_acc = train_correct / data_GT.test_mask.double().sum().item()    
    train_correct = float (pred[data_GT.train_mask].eq(data_GT.y[data_GT.train_mask]).sum().item())
    train_acc = train_correct / data_GT.train_mask.double().sum().item()
    print('Train Accuracy: {:.4f}  correct : {}'.format(train_acc, train_correct))

model.eval()
_, pred = model(data, objects).max(dim=1)
correct = float (pred[data_GT.test_mask].eq(data_GT.y[data_GT.test_mask]).sum().item())
acc = correct / data_GT.test_mask.double().sum().item()

# correct = float (pred[data_GT.train_mask].eq(data_GT.y[data_GT.train_mask]).sum().item())
# acc = correct / data_GT.train_mask.double().sum().item()
print('Accuracy: {:.4f}'.format(acc))