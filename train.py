import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import gc, sys, time
from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros
from layers import CEConv
from utlis import get_normed_lapacian, sp_laplacian_expo, get_sin, get_cos, get_filter

# device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='E:/workspace/pythonProject/pytorch_geometric/tmp/Cora', name='Cora',
                    transform=T.NormalizeFeatures())
# dataset = Planetoid(root='/root/tmp/Cora', name='Cora',
#                     transform=T.NormalizeFeatures())
# dataset = Planetoid(root='E:/workspace/pythonProject/pytorch_geometric/tmp/CiteSeer', name='CiteSeer',
#                     transform=T.NormalizeFeatures())
# dataset = Planetoid(root='/root/tmp/PubMed', name='PubMed', transform=T.NormalizeFeatures())

k_filter = 3 + 1
ita = 1e-3


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = CEConv(in_channels=dataset.num_node_features, out_channels=64, K=k_filter,
                              cos_list=cos_list, sin_list=sin_list)
        self.conv2 = CEConv(in_channels=64, out_channels=dataset.num_classes, K=k_filter,
                              cos_list=cos_list, sin_list=sin_list)

    def forward(self, edge_index, edge_weight, x):
        x = self.conv1(edge_index, edge_weight, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.75, training=self.training)  # training -- apply dropout if is True. Default: True
        x = self.conv2(edge_index, edge_weight, x)

        return F.log_softmax(x, dim=1)  # dim=1


if __name__ == '__main__':

    data = dataset[0].to(device)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    edge_index, edge_weight = get_normed_lapacian(edge_index, normalization='sym')  # eigvalues \in [-1, 1]

    cos_list, sin_list = get_filter(edge_index, edge_weight, k_taylor=19, k_filter=k_filter)  # 20 orders

    model = Net().to(device)
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adam([
        {'params': model.conv1.weight, 'weight_decay': 5e-4},
        {'params': model.conv1.bias, 'weight_decay': 5e-4},
        {'params': model.conv1.a, 'weight_decay': 0},
        {'params': model.conv1.b, 'weight_decay': 0},
        {'params': model.conv2.weight, 'weight_decay': 5e-4},
        {'params': model.conv2.bias, 'weight_decay': 5e-4},
        {'params': model.conv2.a, 'weight_decay': 0},
        {'params': model.conv2.b, 'weight_decay': 0}
    ], lr=0.01)

    res = list()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        y_hat = model(edge_index, edge_weight, x)
        l1loss_ab = torch.norm(model.conv1.a, 1) + torch.norm(model.conv1.b, 1) + torch.norm(model.conv2.a, 1) \
                    + torch.norm(model.conv2.b, 1)
        # loss = F.nll_loss(y_hat[data.train_mask], data.y[data.train_mask])
        loss = F.nll_loss(y_hat[data.train_mask], data.y[data.train_mask]) + ita * l1loss_ab
        loss.backward()
        optimizer.step()

        model.eval()
        # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
        # in the given dimension dim. And indices is the index location of each maximum value found (argmax).
        _, pred = model(edge_index, edge_weight, x).max(dim=1)  # model(data)---->torch.Size([2708, 7])

        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        if epoch % 1 == 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss.item(), acc))

        res.append(acc)
    print('Best accuracy is: {:.4f}'.format(max(res)))





