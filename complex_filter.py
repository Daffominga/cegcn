import torch
import sys, gc
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.utils import get_laplacian
from functools import reduce

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
