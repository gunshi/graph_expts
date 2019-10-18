import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
USE_CUDA = True
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import ipdb
from torch_geometric.datasets import Planetoid
import random

# introduce a form of recurrence within this
# make sure it is symmetric - it is

## not able to propagate labels really

dataset = Planetoid(root='/tmp/Cora', name='Cora')
dataset_GT = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset_GT[0].to('cuda:0')
num_classes = 7
object_model_dim = 256
relation_model_dim = 256
effect_dim = 256
give_only_feats = True  
give_feats = True  ## always keep true
corrupt = True
if(corrupt):
	p = 0.3             
	num_incorrect = 0
	for i in range(dataset[0].y.shape[0]):
	    if(random.uniform(0, 1)<p):
	        r = list(range(0,dataset[0].y[i])) + list(range(dataset[0].y[i]+1, num_classes))
	        dataset[0].y[i] = random.choice(r)
	        num_incorrect += 1


n_objects  = dataset[0].y.shape[0] 
# number of nodes

n_relations  = len(dataset[0].edge_index)  
# number of edges

relation_dim = 1                             

batch_size = 1

y_onehot = torch.zeros((n_objects, num_classes))
y_onehot.scatter_(1, dataset[0].y.unsqueeze(1), 1)                                       

y_onehot_GT = torch.zeros((n_objects, num_classes))
y_onehot_GT.scatter_(1, dataset_GT[0].y.unsqueeze(1), 1)   

target = y_onehot_GT 
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


# receiver_relations, sender_relations - onehot encoding matrices
# each column indicates the receiver and sender object’s index

receiver_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float);
sender_relations   = np.zeros((batch_size, n_objects, n_relations), dtype=float);

cnt = 0

for i in range(len(dataset[0].edge_index)):
    receiver_relations[:, dataset[0].edge_index[0][i], cnt] = 1.0
    sender_relations[:, dataset[0].edge_index[1][i], cnt]   = 1.0
    cnt += 1

relation_info = np.zeros((batch_size, n_relations, relation_dim))

sender_relations   = Variable(torch.FloatTensor(sender_relations))
receiver_relations = Variable(torch.FloatTensor(receiver_relations))
relation_info      = Variable(torch.FloatTensor(relation_info))


if USE_CUDA:
    objects            = objects.cuda()
    sender_relations   = sender_relations.cuda()
    receiver_relations = receiver_relations.cuda()
    relation_info      = relation_info.cuda()
    target             = target.cuda()
    


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x

class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),                                      
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),                                      
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),   
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, num_classes]                                            
        '''
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)


class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, object_dim, n_relations, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()
                                                                                                    
        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, relation_model_dim)
        self.object_model     = ObjectModel(object_dim + effect_dim, object_model_dim)
    
    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        senders   = sender_relations.permute(0, 2, 1).bmm(objects.unsqueeze(0))
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects.unsqueeze(0))

        # batch, nrelations, object dim_send, object dim_recv - > scatter gather
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2))  
        
        effect_receivers = receiver_relations.bmm(effects)
        
        predicted = self.object_model(torch.cat([objects.unsqueeze(0), effect_receivers], 2))

        # do something extra here

        
        # (n objects , numclasses) 
        predicted = F.log_softmax(predicted, dim=1)
        return predicted

interaction_network = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)

if USE_CUDA:
    interaction_network = interaction_network.cuda()
    
optimizer = optim.Adam(interaction_network.parameters(), lr=0.001) #, weight_decay=5e-4)
criterion = nn.MSELoss()


n_epoch = 1500                                                 

losses = []

for epoch in range(n_epoch):

    predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
    loss = F.nll_loss(predicted[data.train_mask], data.y[data.train_mask])  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(np.sqrt(loss.item()))

    if(epoch%5==0):
        _, pred = predicted.max(dim=1)
        train_correct = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.double().sum().item()
        train_acc = train_correct / data.train_mask.double().sum().item()

        print(loss.item())
        print('Train Accuracy: {:.4f}  correct : {}'.format(train_acc, train_correct))
        print('Test Accuracy: {:.4f}  correct : {}'.format(acc, correct))