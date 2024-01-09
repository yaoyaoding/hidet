from typing import List, Tuple, Optional
import dataclasses
from hidet.runtime.compiled_app import CompiledApp, AppMetaData
from hidet.ir.type import data_type
from hidet.graph.tensor import Tensor

from hidet.models.apps.llm.cache import CacheTable


@dataclasses.dataclass
class Attributes:
    cache_dtype: str
    num_layers: int
    num_heads: int
    head_size: int
    block_size: int


class LLM(CompiledApp):
    def __init__(self, attrs: Attributes, memory_capacity: Optional[int] = None):
        super().__init__()
        self.attributes: Attributes = attrs
        self.cache: CacheTable = CacheTable(
            num_gpu_blocks=self._get_num_cache_blocks(memory_capacity, self.attributes),
            num_heads=attrs.num_heads,
            head_size=attrs.head_size,
            block_size=attrs.block_size,
        )

    def _get_num_cache_blocks(self, memory_capacity: int, attrs: Attributes) -> int:
        element_size = data_type(attrs.cache_dtype).nbytes
        size_per_block = attrs.num_heads * attrs.head_size * attrs.block_size * element_size
        return memory_capacity // size_per_block

    def prefill(
        self,
        input_ids: Tensor,  # [bs, seq_length]
        position_ids: Tensor,  # [bs, seq_length]
    ):
        pass

    def decode(
        self,
        input_ids: Tensor,  # [bs, seq_length]
        position_ids: Tensor,  # [bs, seq_length]
    ):
        pass
