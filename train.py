import torch
import torch.nn.functional as F
from layers import ChebConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from complex_filter import getFilter, getNormLapacian
import gc, sys, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='E:/workspace/pythonProject/pytorch_geometric/tmp/Cora', name='Cora',
                    transform=T.NormalizeFeatures())
# dataset = Planetoid(root='E:/workspace/pythonProject/pytorch_geometric/tmp/CiteSeer', name='CiteSeer',
#                     transform=T.NormalizeFeatures())
# dataset = Planetoid(root='E:/workspace/pythonProject/pytorch_geometric/tmp/PubMed', name='PubMed',
#                     transform=T.NormalizeFeatures())

