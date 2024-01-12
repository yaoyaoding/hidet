from typing import List, Union

from hidet.ir import IRModule
from hidet.ir.dtypes import float16
from hidet.graph.tensor import Tensor
from hidet.graph.ops.opaque import OpaqueOperator
from hidet.ir.library import tune


class PageAttentionOp(OpaqueOperator):
    def __init__(
        self, query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
    ):
        super().__init__(
            name='page_attention',
            inputs={
                'query': query,
                'seq_lengths': seq_lengths,
                'cache_blocks': cache_blocks,
                'key_cache': key_cache,
                'value_cache': value_cache,
            },
        )

    def symbolic_forward(
        self, query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
    ):
        assert query.dtype == key_cache.dtype == value_cache.dtype, 'Mismatched dtype of query, key, value'
        assert query.dtype in [float16], f'Unsupported dtype: {query.dtype}'
        bs, num_heads, _, head_size = query.shape
        return [self.symbol(shape=[bs, num_heads, 1, head_size], dtype=query.dtype, device=query.device)]

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1, heads_tile=[2, 4, 8, 16], kv_tile=[16, 32])
    def schedule_cuda(
        self,
        heads_tile: int = 32,
        kv_tile: int = 32,
    ) -> IRModule:
        # naive implementation, todo: optimize this kernel
        import hidet
        from hidet.lang import attrs
        from hidet.lang.types import f16

        _query, _seq_lengths, _cache_blocks, _key_cache, _value_cache = self.inputs[0]
        bs, num_heads, _, head_size = self.inputs[0].shape
        max_cache_blocks = _cache_blocks.shape[-1]
        num_blocks, num_kv_heads, head_size, block_size = _key_cache.shape

        assert int(32 % block_size) == 0
        with hidet.script_module() as script_module:

            @hidet.script
            def page_attention_kernel(
                query: f16[bs, num_heads, head_size],
                seq_lengths: f16[bs],
                cache_blocks: f16[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                output: f16[bs, num_heads, head_size],
            ):
                attrs.func_kind = 'cuda_kernel'

            @hidet.script
            def launch(
                query: f16[bs, num_heads, 1, head_size],
                seq_lengths: f16[bs],
                cache_blocks: f16[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                output: f16[bs, num_heads, 1, head_size],
            ):
                attrs.func_kind = 'public'

                page_attention_kernel(query, seq_lengths, cache_blocks, key_cache, value_cache, output)

        return script_module.ir_module()


def page_attention(query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor):
    """
    Page attention.

    Parameters
    ----------
    query: Tensor
        The query tensor. Shape: [bs, num_heads, 1, head_size]

    seq_lengths: Tensor
        The sequence lengths. Shape: [bs]

    cache_blocks: Tensor
        The cache slots. Shape: [bs, max_cache_blocks]

    key_cache: Tensor
        The key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, 1, head_size]
    """
    return PageAttentionOp(query, seq_lengths, cache_blocks, key_cache, value_cache).outputs[0]
