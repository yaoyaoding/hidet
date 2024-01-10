from typing import Tuple, List
from hidet.ir.type import data_type, DataType
from hidet.graph.tensor import Tensor, empty

KVCache = Tuple[Tensor, Tensor]


class CacheTable:
    def __init__(
        self,
        dtype: DataType,
        memory_capacity: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        block_size: int
    ):
        # [num_blocks, num_heads, head_size, block_size]
        self.dtype: DataType = dtype
        self.memory_capacity: int = memory_capacity
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_size: int = head_size
        self.block_size: int = block_size

        self.gpu_cache: List[KVCache] = self.allocate_cache(memory_capacity, "cuda")
        self.cpu_cache: List[KVCache] = self.allocate_cache(memory_capacity, "cpu")

    def _calc_num_blocks(self, memory_capacity: int) -> int:
        element_size: int = data_type(self.dtype).nbytes
        size_per_block: int = self.num_heads * self.head_size * self.block_size * element_size
        return memory_capacity // (size_per_block * 2 * self.num_layers)

    def allocate_cache(self, memory_capacity: int, device: str) -> List[KVCache]:
        num_blocks: int = self._calc_num_blocks(memory_capacity)
        cache: List[KVCache] = []
        for _ in range(self.num_layers):
            key_cache_shape = [num_blocks, self.num_heads, self.head_size, self.block_size]
            value_cache_shape = [num_blocks, self.num_heads, self.head_size, self.block_size]
            cache.append((empty(shape=key_cache_shape, dtype=self.dtype, device=device),
                          empty(shape=value_cache_shape, dtype=self.dtype, device=device)))
        return cache
