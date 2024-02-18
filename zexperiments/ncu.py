# %%
import torch
from triton_page_attn import triton_paged_attn, triton_paged_attnv2, page_attention_vllm

import hidet
from hidet.apps.llm.ops.page_attention import page_attention as hidet_page_attention

bs = 4
num_heads = 12
head_size = 128
seq_len = 4096
block_size = 16
num_blocks = bs * (seq_len // block_size)

query = torch.empty([bs, num_heads, 1, head_size], device='cuda')
seq_lengths = torch.full([bs], seq_len, device='cuda', dtype=torch.int)
cache_blocks = torch.arange(0, num_blocks, 1, device='cuda', dtype=torch.int).reshape([bs, -1])
key_cache = torch.empty([num_blocks, num_heads, head_size, block_size], device='cuda')
val_cache = torch.empty_like(key_cache)

triton_paged_attn(query, seq_lengths, cache_blocks, key_cache, val_cache)
triton_paged_attnv2(query, seq_lengths, cache_blocks, key_cache, val_cache)
page_attention_vllm(query, seq_lengths, cache_blocks, key_cache, val_cache, max_context_len=seq_len)

query = hidet.from_torch(query)
seq_lengths = hidet.from_torch(seq_lengths)
cache_blocks = hidet.from_torch(cache_blocks)
key_cache = hidet.from_torch(key_cache)
val_cache = hidet.from_torch(val_cache)

hidet.option.cache_dir("zpage_attn_ncu")
hidet.option.debug_cache_tuning()
# hidet.utils.clear_cache_dir()
hidet_page_attention(query, seq_lengths, cache_blocks, key_cache, val_cache, max_context_len=seq_len)



