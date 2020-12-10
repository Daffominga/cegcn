import torch
from typing import Optional
from torch_geometric.typing import OptTensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.inits import glorot, zeros
import complex_filter, sys, gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
