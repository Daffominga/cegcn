import torch
import numpy as np
from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros


class CEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, normalization='sym', bias=True,
                 cos_list=None, sin_list=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(CEConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.k_filter = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cos_list = cos_list
        self.sin_list = sin_list

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        self.a = Parameter(torch.Tensor(K))
        self.b = Parameter(torch.Tensor(K))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        torch.nn.init.uniform_(self.a, a=0.5, b=0.51)
        torch.nn.init.uniform_(self.b, a=0.5, b=0.51)
        glorot(self.weight)
        zeros(self.bias)



    # CEGCN without high-pass filter
    def forward(self, edge_index, edge_weight, x, batch: OptTensor = None):

        Filter = (self.a[0] * self.cos_list[0].to(edge_index.device))  # aI-0b = I
        Tx_0 = x
        out = torch.matmul(torch.matmul(Filter, Tx_0), self.weight[0])
        # out = torch.matmul(x, self.weight[0])

        for k in range(1, self.weight.size(0)):
            Tx_1 = self.propagate(edge_index, edge_weight=edge_weight, x=Tx_0, size=None)
            Tx_0 = Tx_1
            out = out + torch.matmul(Tx_1, self.weight[k])

        if self.bias is not None:
            out += self.bias

        return out


    '''
    def forward(self, edge_index, edge_weight, x, batch: OptTensor = None):

        Filter = (self.a[0] * self.cos_list[0].to(edge_index.device))  # aI-0b = I
        Tx_0 = x
        out = torch.matmul(torch.matmul(Filter, Tx_0), self.weight[0])
        # out = torch.matmul(x, self.weight[0])

        # FL^kX
        for k in range(1, self.weight.size(0)):
            Filter = (self.a[k] * self.cos_list[k].to(edge_index.device) -
                      self.b[k] * self.sin_list[k].to(edge_index.device))
            Tx_1 = self.propagate(edge_index, edge_weight=edge_weight, x=Tx_0, size=None)
            Tx_0 = Tx_1
            Tx_2 = torch.matmul(Filter, Tx_1)
            out = out + torch.matmul(Tx_2, self.weight[k])

        if self.bias is not None:
            out += self.bias

        return out
    '''

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels], edge_weight has shape [1, E]
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)
