from typing import List, Dict, Optional
import math
import transformers
from hidet import nn, ops
from hidet.graph.tensor import Tensor


class AttentionState:
    def run(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError()


class DefaultAttnState(AttentionState):
    def __init__(self):
        self.is_prefill: bool = True
        self.key_cache: Optional[Tensor] = None     # [bs, num_kv_heads, seq_length, head_size]
        self.value_cache: Optional[Tensor] = None   # [bs, num_kv_heads, seq_length, head_size]

    def run(self, query: Tensor, key: Tensor, value: Tensor):
        if self.is_prefill:
            # prefill stage
            query = ops.transpose(query, axes=[0, 1, 3, 2])  # [bs, num_heads, head_size, seq_length]
            score = ops.matmul(query, key) / math.sqrt(query.shape[-2])  # [bs, num_heads, seq_length, seq_length]
            seq_length = score.shape[-1]
            tri = ops.tri(seq_length, seq_length, dtype=score.dtype, device=score.device)
            causal_mask = (score.dtype.one - tri) * score.dtype.min_value
            score = ops.softmax(score + causal_mask, axis=-1)
            output = ops.matmul(score, value)
            
            self.key_cache = key
            self.value_cache = value
            return output
        else:
            # decode stage
            # key, query: [bs, num_heads, past_length, head_size]
            key = ops.concat([self.key_cache, key], axis=-2)
            value = ops.concat([self.value_cache, value], axis=-2)
            query = ops.transpose(query, axes=[0, 1, 3, 2])
            score = ops.matmul(query, key) / math.sqrt(value.shape[-1])  # [num_heads, seq_length, past_length]
            seq_length = score.shape[-2]
            past_length = score.shape[-1]
            tri = ops.tri(seq_length, seq_length + past_length, k=past_length, dtype=score.dtype, device=score.device)
            causal_mask = (score.dtype.one - tri) * score.dtype.min_value
            score = ops.softmax(score + causal_mask, axis=-1)
            output = ops.matmul(score, value)
            
            self.key_cache = key
            self.value_cache = value
            return output


class PagedAttnState(AttentionState):
    def __init__(self, is_prefill: bool, key_cache: Tensor, value_cache: Tensor, cache_slots: Tensor):
        self.is_prefill: bool = is_prefill
        self.key_cache: Optional[Tensor] = key_cache      # [num_blocks, num_heads, head_size, block_size]
        self.value_cache: Optional[Tensor] = value_cache  # [num_blocks, num_heads, head_size, block_size]
        self.cache_slots: Optional[Tensor] = cache_slots  # [bs, seq_length]

    def store_cache(self, key: Tensor, value: Tensor):
        pass

    def run(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        pass


class Attention(nn.Module):
    def forward(
        self,
        query: Tensor,  # [bs, num_heads, seq_length, head_size]
        key: Tensor,    # [bs, num_kv_heads, seq_length, head_size]
        value: Tensor,  # [bs, num_kv_heads, seq_length, head_size]
        state: AttentionState
    ):
        return state.run(query, key, value)
