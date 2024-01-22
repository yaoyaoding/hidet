# %%
import torch
from torch import Tensor

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
        The key cache. Shape: [num_blocks, num_heads, head_size, block_size]

    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_heads, head_size, block_size]

    Returns
    -------
    output: Tensor
        The output tensor. Shape: [bs, num_heads, 1, head_size]
    """
    result = []
    _, num_heads, head_size, block_size = key_cache.shape
    for bi, seq in enumerate(seq_lengths):
        maxes = torch.full([num_heads], float('-inf'), device=query.device)
        sums = torch.zeros([num_heads], device=query.device)
        acc = torch.zeros([num_heads, head_size], device=query.device)
        for block_pos in range(0, (int(seq) + block_size - 1) // block_size):
            cache_idx = cache_blocks[bi, block_pos]

            qk = (query[bi] @ key_cache[cache_idx]).squeeze() # [num_heads, block_size]
            new_maxes = torch.maximum(torch.max(qk, 1).values, maxes)
            qk = torch.exp(qk - new_maxes[:, None])
            sums = sums * (maxes / new_maxes).exp() + qk.sum(1)
            maxes = new_maxes
            

            qkv = qk.unsqueeze(1) @ value_cache[cache_idx].transpose(2, 1) # [num_heads, head_size]
            acc += qkv.squeeze()
        acc = acc / sums[:, None]
        result.append(acc)
    return torch.stack(result).unsqueeze(2)

import triton
import triton.language as tl

from triton.ops.flash_attention import _fwd_kernel
@triton.jit
def page_attn_kernel(
    out, # f32/f16[bs, num_heads, 1, head_size]
    query, # f32/f16[bs, num_heads, 1, head_size]
    seq_lens, # i32[bs]
    cache_blocks, # i32[bs, max_cache_blocks]
    key_cache, # f32/f16[num_blocks, num_heads, head_size, block_size]
    val_cache, # f32/f16[num_blocks, num_heads, head_size, block_size]
    num_heads,
    max_cache_blocks,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
):
    bi = tl.program_id(0) // num_heads
    hi = tl.program_id(0) % num_heads

    seq_len = tl.load(seq_lens + bi)

    h_dim = tl.arange(0, head_size)
    seq_dim = tl.arange(0, block_size)

    query_ptrs = query + bi * num_heads * head_size + hi * head_size + h_dim
    query = tl.load(query_ptrs) # [head_size]
    
    kv_offsets = h_dim[:, None] * block_size + seq_dim[None, :]

    mi = float('-inf')
    si = 0.0
    acc = tl.zeros([head_size], dtype=key_cache.dtype.element_ty)

    for i in range(0, tl.cdiv(seq_len, block_size)):
        block_idx = tl.load(cache_blocks + bi * max_cache_blocks + i)
        kv_offset = block_idx * num_heads * head_size * block_size + hi * head_size * block_size + kv_offsets
        key = tl.load(key_cache + kv_offset) # [head_size, block_size]
        qk = tl.sum(query[:, None] * key, 0) # [block_size]
        new_mi = tl.maximum(tl.max(qk, 0), mi)
        qk = tl.exp(qk - new_mi)
        si = si * tl.exp(mi / new_mi) + tl.sum(qk, 0)
        mi = new_mi
        val = tl.load(val_cache + kv_offset) # [head_size, block_size]
        qkv = tl.sum(qk[None, :] * val, 1) # [head_size]
        acc += qkv
    
    acc /= si

    out_offset = bi * num_heads * head_size + hi * head_size + tl.arange(0, head_size)
    tl.store(out + out_offset, acc)


def triton_paged_attn(query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor):
    bs, num_heads, _, head_size = query.shape
    _, max_cache_blocks = cache_blocks.shape
    _, num_heads, _, block_size = key_cache.shape
    out = torch.empty_like(query)

    page_attn_kernel[(bs, num_heads)](out, query, seq_lengths, cache_blocks, key_cache,
                                      value_cache, num_heads, max_cache_blocks, 
                                      head_size, block_size)

    return out

def make_inputs(bs=4, num_heads=8, head_size=4, seq_len=64, block_size=16):
    num_blocks = bs * seq_len // block_size

    query = torch.randn([bs, num_heads, 1, head_size], device='cuda')
    seq_lengths = torch.full([bs], seq_len, device='cuda', dtype=torch.int)
    cache_blocks = torch.arange(0, num_blocks, 1, device='cuda', dtype=torch.int).reshape([bs, -1])
    key_cache = torch.randn([num_blocks, num_heads, head_size, block_size], device='cuda')
    val_cache = torch.randn_like(key_cache)
    return query, seq_lengths, cache_blocks, key_cache, val_cache
from vllm.model_executor.layers.attention import _paged_attention
from vllm.model_executor import InputMetadata

def page_attention_vllm(query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor, max_context_len: int = 1024):
    """
    query: [bs, num_heads, 1, head_size]
    key_cache: [num_blocks, num_heads, head_size, block_size]
    value_cache: [num_blocks, num_heads, head_size, block_size]
    """
    bs, nh, _, hs = query.shape
    x = 16 // torch.tensor([], dtype=query.dtype).element_size()
    num_blocks, num_heads, head_size, block_size = value_cache.shape
    # key_cache: [num_blocks, num_heads, head_size//x, block_size, x]
    key_cache = key_cache.view(num_blocks, num_heads, head_size // x, block_size, x)
    return _paged_attention(
        query.view(bs, nh, hs), key_cache, value_cache, 
        InputMetadata(False, None, max_context_len, seq_lengths, cache_blocks, False), num_heads, 1.0, None
    )


def test():
    bs = 4
    num_heads = 8
    head_size = 64
    seq_len = 64
    block_size = 16
    num_blocks = bs * seq_len // block_size

    query = torch.randn([bs, num_heads, 1, head_size], device='cuda')
    seq_lengths = torch.full([bs], seq_len, device='cuda', dtype=torch.int)
    cache_blocks = torch.arange(0, num_blocks, 1, device='cuda', dtype=torch.int).reshape([bs, -1])
    key_cache = torch.randn([num_blocks, num_heads, head_size, block_size], device='cuda')
    val_cache = torch.randn_like(key_cache)

    new_shape = [bs, seq_len // block_size, num_heads, head_size, block_size]
    orig_shape = [bs, num_heads, head_size, seq_len]
    qk0 = (query @ key_cache.reshape(new_shape).permute(0, 2, 3, 1, 4).reshape(orig_shape)).softmax(-1)
    out0 = qk0 @ val_cache.reshape(new_shape).permute(0, 2, 1, 4, 3).reshape([bs, num_heads, seq_len, head_size])

    out1 = page_attention(query, seq_lengths, cache_blocks, key_cache, val_cache)

    print((out0 - out1).abs().max())

    out2 = triton_paged_attn(query, seq_lengths, cache_blocks, key_cache, val_cache)

    print((out1 - out2).abs().max())

    x = 16 // torch.tensor([], dtype=query.dtype).element_size()
    key_cache_ = key_cache.reshape([num_blocks, num_heads, head_size//x, x, block_size]).transpose(-1, -2).reshape([num_blocks, num_heads, head_size, block_size])
    out3 = page_attention_vllm(query, seq_lengths, cache_blocks, key_cache_, val_cache)
    print((out0.squeeze() - out3).abs().max())

test()

# %%

bs = 4
num_heads = 8
head_size = 128
block_size = 32

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 13)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['vllm', 'triton'],
        # Label name for the lines
        line_names=["vllm", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="ms",  # Label name for the y-axis
        plot_name=f"bs={bs},nheads={num_heads},hsize={head_size},bsize={block_size}",
        args={},
    ))
def benchmark(seq, provider):
    args = make_inputs(bs, num_heads, head_size, seq, block_size)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'vllm':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: page_attention_vllm(*args, max_context_len=seq), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_paged_attn(*args), quantiles=quantiles)
    # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True)
