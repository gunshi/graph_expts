from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])



dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6))


# whaaa
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6),
                    transform=T.RandomTranslate(0.01))


dataset[0]