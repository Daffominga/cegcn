import torch, numpy
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from functools import reduce


# get normed eigenvalues in [-1, 1]
def get_normed_lapacian(edge_index, normalization: Optional[str], lambda_max=None,
                        edge_weight: OptTensor = None, dtype: Optional[int] = None, batch: OptTensor = None):

    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization)

    if batch is not None and lambda_max.numel() > 1:
        lambda_max = lambda_max[batch[edge_index[0]]]
    else:
        L = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        lambda_max = max(torch.eig(L)[0][:, 0])

    edge_weight = (2.0 * edge_weight) / lambda_max
    edge_weight.masked_fill_(edge_weight == float('inf'), 0)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1)

    assert edge_weight is not None

    return edge_index, edge_weight


def sp_laplacian_expo(edge_index, edge_weight, expo):
    num_nodes = maybe_num_nodes(edge_index)

    index_tmp = edge_index.clone()
    print(index_tmp.device)
    value_tmp = edge_weight.clone()
    for i in range(1, expo):
        index_tmp, value_tmp = spspmm(index_tmp, value_tmp, edge_index, edge_weight,
                                      num_nodes, num_nodes, num_nodes, coalesced=True)

    return index_tmp, value_tmp


def get_cos(edge_index, edge_weight, k_taylor):
    L = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    L2 = torch.matmul(L, L)
    L_tmp = L2.clone()
    cos = torch.eye(L.shape[0], device=L.device)  # cos(0) = I
    cos -= L2 / torch.tensor(data=[2.], device=L.device)

    for i in range(4, k_taylor, 2):
        factorial = reduce(lambda x, y: x * y, range(1, i + 1))
        L_tmp = torch.matmul(L_tmp, L2)
        cos += (((-1) ** (i / 2)) * L_tmp) / factorial

    return cos


def get_sin(edge_index, edge_weight, k_taylor):
    L = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    L2 = torch.matmul(L, L)
    L_tmp = L.clone()
    sin = L.clone()

    for i in range(3, k_taylor + 1, 2):
        factorial = reduce(lambda x, y: x * y, range(1, i + 1))
        L_tmp = torch.matmul(L_tmp, L2)
        sin += (((-1) ** ((i - 1) / 2)) * L_tmp) / factorial

    return sin


def get_filter(edge_index, edge_weight, k_taylor, k_filter):
    edge_weight = edge_weight
    num_nodes = maybe_num_nodes(edge_index)
    cos_list = list()
    sin_list = list()
    cos = get_cos(edge_index, edge_weight, k_taylor=k_taylor)
    sin = get_sin(edge_index, edge_weight, k_taylor=k_taylor)

    cos_list.append(torch.eye(n=num_nodes))
    sin_list.append(torch.zeros(size=(num_nodes, num_nodes)))

    for i in range(1, k_filter):
        if i == 1:
            cos_list.append(cos)
            sin_list.append(sin)
        else:
            cos_list.append(cos_list[-1] @ cos - sin_list[-1] @ sin)
            sin_list.append(cos_list[-1] @ sin + sin_list[-1] @ cos)

    return cos_list, sin_list

