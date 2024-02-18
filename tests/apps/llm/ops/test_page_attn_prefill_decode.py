# %%
import pytest

import torch
import hidet

hidet.option.cache_dir('test_page_attn2')
# hidet.utils.clear_cache_dir()
def attn(q, k, v):
    qk = q @ k.transpose(-2, -1)
    qk = qk / q.size(-1)**0.5
    qk = torch.nn.functional.softmax(qk, dim=-1)
    return qk @ v

def convert_kcache1(k):
    max_num_blocks, num_kv_heads, head_size, block_size = k.shape
    x = 16 // torch.tensor([], dtype=k.dtype).element_size()
    k1 = k.reshape(max_num_blocks, num_kv_heads, head_size // x, x, block_size)
    k1 = k1.permute(0, 1, 2, 4, 3).reshape(max_num_blocks, num_kv_heads, head_size, block_size)
    return k1

def kv_cache_to_page_attn_cache(x: torch.Tensor, block_size=16):
    # x: [bs, num_heads, seq_len, head_size] -> [num_blocks, num_heads, head_size, block_size]
    bs, num_heads, seq_len, head_size = x.shape
    seq_rup = (seq_len + block_size - 1) // block_size * block_size
    x = torch.nn.functional.pad(x, [0, 0, 0, seq_rup - seq_len, 0, 0, 0, 0], mode='constant', value=0)
    xs = x.permute(0, 2, 1, 3)\
        .reshape(-1, block_size, num_heads, head_size).permute(0, 2, 3, 1).contiguous()
    return torch.nn.functional.pad(xs, [0, 0, 0, 0, 0, 0, 0, 3], mode='constant', value=0)


@pytest.mark.parametrize('test_prefill', [True, False])
@pytest.mark.parametrize('seq_len', [16, 32, 65])
@pytest.mark.parametrize('num_heads', [8, 32])
@pytest.mark.parametrize('head_size', [32, 64, 128])
@pytest.mark.parametrize('block_size', [16, 32])
def test_decode(test_prefill, seq_len, num_heads, head_size, block_size):
    from hidet.apps.llm.nn.attention import page_attention, cache_write
    batch_size = 1

    q =  torch.randn([batch_size, num_heads, 1, head_size], dtype=torch.float32).cuda()
    k1 = torch.randn([batch_size, num_heads, 1, head_size], dtype=torch.float32).cuda()
    v1 = torch.randn([batch_size, num_heads, 1, head_size], dtype=torch.float32).cuda()

    kc = torch.randn([batch_size, num_heads, seq_len, head_size]).cuda()
    vc = torch.randn([batch_size, num_heads, seq_len, head_size]).cuda()

    kc1 = torch.cat([kc, k1], dim=-2)
    vc1 = torch.cat([vc, v1], dim=-2)
    y1 = attn(q, kc1, vc1)

    kcache = convert_kcache1(kv_cache_to_page_attn_cache(kc, block_size))
    vcache = kv_cache_to_page_attn_cache(vc, block_size)
    if test_prefill:
        kcache = torch.empty_like(kcache)
        vcache = torch.empty_like(vcache)
        slots = hidet.asarray([list(range(0, seq_len))], dtype='int64', device='cuda') 
        seq_lengths = hidet.from_torch(torch.full([1], seq_len, dtype=torch.int32, device='cuda'))
        kcache, vcache = cache_write(seq_lengths, hidet.from_torch(kc), hidet.from_torch(vc), slots, hidet.from_torch(kcache), hidet.from_torch(vcache))
        kcache = kcache.torch()
        vcache = vcache.torch()
    
    blocks = hidet.from_torch(torch.arange(0, kcache.shape[0], 1, dtype=torch.int32, device='cuda').reshape(1, -1))
    seq_lengths = hidet.from_torch(torch.full([1], seq_len + 1, dtype=torch.int32, device='cuda'))
    slots = hidet.asarray([[seq_len]], dtype='int64', device='cuda')



    kcache1, vcache1 = cache_write(seq_lengths, hidet.from_torch(k1), hidet.from_torch(v1), slots, 
                                hidet.from_torch(kcache.clone()), hidet.from_torch(vcache.clone()), )


    y2 = page_attention(hidet.from_torch(q), seq_lengths, blocks, kcache1, vcache1)
    y2 = y2.torch()
    print((y1 - y2).max().abs())

    assert torch.allclose(y1, y2, atol=1e-3, rtol=1e-3)

# test_decode(False, 65, 8, 128, 16)

if __name__ == '__main__':
    pytest.main([__file__])
