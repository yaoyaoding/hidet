# %%
import pytest
import random
import torch

import hidet.utils
import hidet.option
from hidet import Tensor, from_torch
from hidet.apps.llm.ops.page_attention import page_attention, cache_write

def convert_kcache1(k):
    max_num_blocks, num_kv_heads, head_size, block_size = k.shape
    x = 16 // torch.tensor([], dtype=k.dtype).element_size()
    k1 = k.reshape(max_num_blocks, num_kv_heads, head_size // x, x, block_size)
    k1 = k1.permute(0, 1, 2, 4, 3).reshape(max_num_blocks, num_kv_heads, head_size, block_size)
    return k1


def kv_cache_to_page_attn_cache(x: torch.Tensor, block_size=16):
    # x: [bs, num_heads, seq_len, head_size] -> [num_blocks, num_heads, head_size, block_size]
    bs, num_heads, seq_len, head_size = x.shape
    xs = x.permute(0, 2, 1, 3)\
        .reshape(-1, block_size, num_heads, head_size).permute(0, 2, 3, 1).contiguous()
    return torch.nn.functional.pad(xs, [0, 0, 0, 0, 0, 0, 0, 1], mode='constant', value=0)

def cache_write_ref(
    seq_lengths: Tensor, key: Tensor, value: Tensor, cache_slots: Tensor, key_cache: Tensor, value_cache: Tensor
):
    # convert to torch tensors
    seq_lengths = seq_lengths.torch()
    key = key.torch()
    value = value.torch()
    cache_slots = cache_slots.torch()
    key_cache = key_cache.torch().clone()
    value_cache = value_cache.torch().clone()

    bs = seq_lengths.size(0)
    block_size = key_cache.size(-1)

    is_prefill = key.size(2) > 1

    for i in range(bs):
        if is_prefill:
            seq_length = seq_lengths[i]
            cache_slots_list = cache_slots[i, :seq_length].tolist()
            token_list = list(range(seq_length))
        else:
            cache_slots_list = cache_slots[i, :1].tolist()
            token_list = [0]
        for token_idx, cache_slot in zip(token_list, cache_slots_list):
            block_idx = cache_slot // block_size
            slot_idx = cache_slot % block_size
            print(block_idx, slot_idx)

            key_cache[block_idx, :, :, slot_idx] = key[i, :, token_idx, :]
            value_cache[block_idx, :, :, slot_idx] = value[i, :, token_idx, :]

    key_cache = convert_kcache1(key_cache)
    return from_torch(key_cache), from_torch(value_cache)


@pytest.mark.parametrize('num_kv_heads', [1, 32])
@pytest.mark.parametrize('block_size', [16, 32])
@pytest.mark.parametrize('head_size', [128])
@pytest.mark.parametrize('seq_lengths_list', [[2, 3, 999, 3282, 18, 1, 32]])
@pytest.mark.parametrize('is_prefill', [True, False])
def test_cache_write(num_kv_heads, block_size, head_size, seq_lengths_list, is_prefill):

    bs = len(seq_lengths_list)
    max_seq_length = max(seq_lengths_list)
    cache_slots_list = []
    current_slot = 0
    for seq_length in seq_lengths_list:
        slots = []
        for i in range(seq_length):
            slots.append(current_slot)
            current_slot += 1
        # random.shuffle(slots)
        slots = list(reversed(slots))
        if not is_prefill:
            slots = [slots[-1]]
        else:
            while len(slots) < max_seq_length:
                slots.append(-1)
        current_slot = (current_slot + block_size - 1) // block_size * block_size
        cache_slots_list.append(slots)
    num_blocks = (current_slot + block_size - 1) // block_size

    seq_lengths = from_torch(torch.asarray(seq_lengths_list, dtype=torch.int32, device='cuda'))
    key = from_torch(
        torch.randn(
            bs, num_kv_heads, max_seq_length if is_prefill else 1, head_size, dtype=torch.float16, device='cuda'
        )
    )
    # key = from_torch(torch.ones(bs, num_kv_heads, max_seq_length, head_size, dtype=torch.float16, device='cuda'))
    value = from_torch(
        torch.randn(
            bs, num_kv_heads, max_seq_length if is_prefill else 1, head_size, dtype=torch.float16, device='cuda'
        )
    )
    cache_slots = from_torch(torch.asarray(cache_slots_list, dtype=torch.int64, device='cuda'))
    key_caches = [
        from_torch(torch.zeros(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16, device='cuda'))
        for _ in range(2)
    ]
    value_caches = [
        from_torch(torch.zeros(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16, device='cuda'))
        for _ in range(2)
    ]

    # print('key: ', key)
    # print('value: ', value)
    # print('cache_slots: ', cache_slots)
    # print('key_caches: ', key_caches)
    # print('value_caches: ', value_caches)

    for idx, func in enumerate([cache_write, cache_write_ref]):
        key_cache, value_cache = func(seq_lengths, key, value, cache_slots, key_caches[idx], value_caches[idx])
        key_caches[idx] = key_cache
        value_caches[idx] = value_cache

    # print(key_caches[0])
    # print(key_caches[1])
    # print((key_caches[0].torch() - key_caches[1].torch()).abs().max())
    # print((value_caches[0].torch() - value_caches[1].torch()).abs().max())
    # hidet.utils.assert_close(key_caches[0], key_caches[1])
    # hidet.utils.assert_close(value_caches[0], value_caches[1])
    return key_caches[0].torch(), key_caches[1].torch()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_cache_write_2(dtype):
    # hidet.utils.clear_cache_dir()
    k1 = torch.randn([1, 4, 1, 16], dtype=dtype).cuda()
    v1 = torch.randn([1, 4, 1, 16], dtype=dtype).cuda()

    kc = torch.randn([1, 4, 16, 16], dtype=dtype).cuda()
    vc = torch.randn([1, 4, 16, 16], dtype=dtype).cuda()

    seq_lens = torch.tensor([15], dtype=torch.int32, device='cuda')
    slots = torch.tensor([[15]], dtype=torch.int64, device='cuda')

    kc1 = convert_kcache1(kv_cache_to_page_attn_cache(torch.cat([kc[:, :, :15], k1], dim=-2)))
    vc1 = kv_cache_to_page_attn_cache(torch.cat([vc[:, :, :15], v1], dim=-2))

    f = lambda x: hidet.from_torch(x)

    kcache = convert_kcache1(kv_cache_to_page_attn_cache(kc))
    vcache = kv_cache_to_page_attn_cache(vc)
    # kc2, vc2 = cache_write_ref(f(seq_lens), f(k1), f(v1), f(slots), f(kcache), f(vcache))
    kc2, vc2 = cache_write(f(seq_lens), f(k1), f(v1), f(slots), f(kcache.clone()), f(vcache.clone()))
    kc2 = kc2.torch()
    vc2 = vc2.torch()
    # print(vc1.shape, vc2.shape)
    # print((kc1[0, :, :, :15] - kc2[0, :, :, :15]).abs().max())
    # print((vc1 - vc2).abs().max())
    # print((kc1 - kc2).abs().max())
    assert torch.allclose(kc1, kc2)
    assert torch.allclose(vc1, vc2)

test_cache_write_2(torch.float16)

if __name__ == '__main__':
    pytest.main([__file__])

