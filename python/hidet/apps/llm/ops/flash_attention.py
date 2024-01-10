from hidet.graph.tensor import Tensor
from hidet.graph.ops.attention import attention as _attention


def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor
) -> Tensor:
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
    return _attention(query, key, value, is_causal=True)
