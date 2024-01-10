
def page_attention(
    query,
    key,
    value,
    cache_blocks,
    key_cache,
    value_cache
):
    """
    Page attention.

    Parameters
    ----------
    query: Tensor
        The query tensor. Shape: [bs, num_heads, 1, head_size]

    key: Tensor
        The key tensor. Shape: [bs, num_kv_heads, 1, head_size]

    value: Tensor
        The value tensor. Shape: [bs, num_kv_heads, 1, head_size]

    cache_blocks: Tensor
        The cache slots. Shape: [bs, max_cache_blocks]

    key_cache: Tensor
        The key cache. Shape: [num_blocks, num_heads, head_size, block_size]

    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_heads, head_size, block_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, 1, head_size]
    """
    pass
