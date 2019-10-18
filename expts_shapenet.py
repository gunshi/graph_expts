from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
import ipdb


# seed all the randomness still once here though

# load data

dataset1 = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

dataset2 = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])


# dataset1 = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6))


# # whaaa
# dataset2 = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6),
#                     transform=T.RandomTranslate(0.01))


ipdb.set_trace()

dataset1[0]

# (pointconv is sota)