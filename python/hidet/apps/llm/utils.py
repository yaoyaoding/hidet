from typing import List, Any
import torch
from hidet.graph.tensor import Tensor, from_dlpack


def tensor_pad(
    data: List[List[Any]],
    max_length: int,
    pad_value: Any = 0,
    dtype: str = 'int32',
    device: str = 'cuda'
) -> Tensor:
    data = [row + [pad_value] * (max_length - len(row)) for row in data]
    return from_dlpack(torch.tensor(data, dtype=getattr(torch, dtype), device=device))
