import math
from hidet.graph.tensor import Tensor
from hidet.graph.ops.attention import attention as _attention
from hidet.graph import ops


def flash_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Flash attention.

    Parameters
    ----------
    query: Tensor
        The query tensor. Shape: [bs, num_heads, seq_length, head_size]

    key: Tensor
        The key tensor. Shape: [bs, num_kv_heads, seq_length, head_size]

    value: Tensor
        The value tensor. Shape: [bs, num_kv_heads, seq_length, head_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, seq_length, head_size]
    """
    # return _attention(query, key, value, is_causal=True)
    from hidet.ir.expr import cast

    key = ops.transpose(key, axes=[0, 1, 3, 2])  # [bs, num_heads, head_size, seq_length]
    # [1, num_heads, seq_length, seq_length]
    score = ops.matmul(query, key) / math.sqrt(query.shape[-2])
    seq_length = score.shape[-1]
    tri = ops.tri(seq_length, seq_length, dtype=score.dtype, device=score.device)
    causal_mask = (1.0 - tri) * score.dtype.min_value
    score = ops.softmax(score + causal_mask, axis=-1)
    output = ops.matmul(score, value)

    return output
