from typing import Tuple, List
from hidet.graph.tensor import Tensor


KVCache = Tuple[Tensor, Tensor]


class CacheTable:
    def __init__(self, num_gpu_blocks: int, num_heads: int, head_size: int, block_size: int):
        # [num_blocks, num_heads, head_size, block_size]
        self.gpu_cache: List[KVCache] = self.allocate_gpu_cache(num_gpu_blocks, num_heads, head_size, block_size)

    def allocate_gpu_cache(self, num_gpu_blocks, num_heads, head_size, block_size) -> List[KVCache]:
        pass
