# %%
from typing import List, Union, Tuple

from hidet.ir import IRModule
from hidet.ir.dtypes import float16, float32
from hidet.ir.expr import symbol_var
from hidet.graph.tensor import Tensor
from hidet.graph.ops.opaque import OpaqueOperator
from hidet.ir.library import tune


class PageAttentionWriteCacheOp(OpaqueOperator):
    def __init__(self, seq_lengths, key, value, cache_slots, key_cache, value_cache):
        # seq_lengths: [bs]
        #   key: [bs, num_kv_heads, max_seq_length, head_size]
        # value: [bs, num_kv_heads, max_seq_length, head_size]
        # cache_slots: [bs, max_seq_length]
        # key_cache: [num_blocks, num_kv_heads, head_size, block_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        super().__init__(
            name='cache_write',
            inputs={
                'seq_lengths': seq_lengths,
                'key': key,
                'value': value,
                'cache_slots': cache_slots,
                'key_cache': key_cache,
                'value_cache': value_cache,
            },
            share_map={0: 4, 1: 5},  # share key_cache and value_cache
        )

    def symbolic_forward(self, seq_lengths, key, value, cache_slots, key_cache, value_cache):
        return {'key_cache_out': key_cache, 'value_cache_out': value_cache}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)  # empty space
    def schedule_cuda(self):
        import hidet
        from hidet.lang import attrs, printf
        from hidet.lang.types import i32, f16, f32, i64, shared_tensor
        from hidet.lang.cuda import blockIdx, threadIdx, blockDim, syncthreads
        from hidet.lang.mapping import spatial

        bs, num_kv_heads, max_seq_length, head_size = self.inputs[1].shape
        num_blocks, num_kv_heads, head_size, block_size = self.inputs[4].shape

        with hidet.script_module() as script_module:
            seq_tile = 1
            dim_tile = 1
            assert int(head_size % (dim_tile * 4)) == 0
            assert int(block_size % (seq_tile * 4)) == 0

            @hidet.script
            def _prefill_cache_write(
                seq_lengths: i32[bs],
                inp: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                buf: f16[seq_tile, dim_tile * 4 + 1],
            ):
                attrs.func_kind = 'cuda_internal'

                bs_idx = blockIdx.x
                kv_head_idx = blockIdx.y
                seq_length = seq_lengths[bs_idx]

                for t in range((seq_length + seq_tile - 1) // seq_tile):
                    if (t + 1) * seq_tile < seq_length:
                        # do not need to check boundary
                        for j in range(head_size // (dim_tile * 4)):
                            # read to buf [seq_tile, dim_tile * 4]
                            for i, jj in spatial(seq_tile, dim_tile).on(threadIdx.x):
                                for jjj in range(4):
                                    seq_idx = t * seq_tile + i
                                    dim_idx = j * (dim_tile * 4) + jj * 4 + jjj
                                    buf[i, jj * 4 + jjj] = inp[bs_idx, kv_head_idx, seq_idx, dim_idx]
                            syncthreads()

                            # write buf to cache in global memory
                            for jj, ii in spatial(dim_tile, seq_tile).on(threadIdx.x):
                                seq_idx = t * seq_tile + ii
                                cache_slot = cache_slots[bs_idx, seq_idx]
                                block_idx = i32(cache_slot // block_size)
                                block_offset = i32(cache_slot % block_size)
                                for jjj in range(4):
                                    dim_idx = j * dim_tile * 4 + jj * 4 + jjj
                                    cache[block_idx, kv_head_idx, dim_idx, block_offset] = buf[ii, jj * 4 + jjj]
                            syncthreads()
                    else:
                        # do not need to check boundary
                        for j in range(head_size // (dim_tile * 4)):
                            # read to buf [seq_tile, dim_tile * 4]
                            for i, jj in spatial(seq_tile, dim_tile).on(threadIdx.x):
                                for jjj in range(4):
                                    seq_idx = t * seq_tile + i
                                    dim_idx = j * (dim_tile * 4) + jj * 4 + jjj
                                    if seq_idx < seq_length:
                                        buf[i, jj * 4 + jjj] = inp[bs_idx, kv_head_idx, seq_idx, dim_idx]
                            syncthreads()

                            # write buf to cache in global memory
                            for jj, ii in spatial(dim_tile, seq_tile).on(threadIdx.x):
                                seq_idx = t * seq_tile + ii
                                if seq_idx < seq_length:
                                    cache_slot = cache_slots[bs_idx, seq_idx]
                                    block_idx = i32(cache_slot // block_size)
                                    block_offset = i32(cache_slot % block_size)
                                    for jjj in range(4):
                                        dim_idx = j * dim_tile * 4 + jj * 4 + jjj
                                        cache[block_idx, kv_head_idx, dim_idx, block_offset] = buf[ii, jj * 4 + jjj]
                            syncthreads()

            @hidet.script
            def prefill_cache_write(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, max_seq_length, head_size],
                value: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = bs, num_kv_heads
                attrs.cuda.block_dim = seq_tile * dim_tile

                buf = shared_tensor(dtype=f16, shape=[seq_tile, dim_tile * 4 + 1])

                _prefill_cache_write(seq_lengths, key, cache_slots, key_cache, buf)
                _prefill_cache_write(seq_lengths, value, cache_slots, value_cache, buf)

            @hidet.script
            def decode_cache_write(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, 1, head_size],
                value: f16[bs, num_kv_heads, 1, head_size],
                cache_slots: i64[bs, 1],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = bs, num_kv_heads
                attrs.cuda.block_dim = head_size

                bs_idx = blockIdx.x
                kv_head_idx = blockIdx.y
                dim_idx = threadIdx.x

                # get cache slot
                cache_slot = cache_slots[bs_idx, 0]
                block_idx = cache_slot / block_size
                block_offset = cache_slot % block_size

                # store key and value to cache
                key_cache[block_idx, kv_head_idx, dim_idx, block_offset] = key[bs_idx, kv_head_idx, 0, dim_idx]
                value_cache[block_idx, kv_head_idx, dim_idx, block_offset] = value[bs_idx, kv_head_idx, 0, dim_idx]

            @hidet.script
            def launch(
                seq_lengths: i32[bs],
                key: f16[bs, num_kv_heads, max_seq_length, head_size],
                value: f16[bs, num_kv_heads, max_seq_length, head_size],
                cache_slots: i64[bs, max_seq_length],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                key_cache_out: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache_out: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'public'

                if max_seq_length == 1:
                    decode_cache_write(seq_lengths, key, value, cache_slots, key_cache, value_cache)
                else:
                    prefill_cache_write(seq_lengths, key, value, cache_slots, key_cache, value_cache)

        return script_module.ir_module()


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
        return {'output': self.symbol(shape=[bs, num_heads, 1, head_size], dtype=query.dtype, device=query.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1)  # empty space
    def schedule_cuda(self) -> IRModule:
        # naive implementation, todo: optimize this kernel
        import hidet
        from hidet.lang import attrs, cast, printf
        from hidet.lang.types import u8, f16, f32, i32, register_tensor, tensor_pointer_type, tensor_pointer
        from hidet.lang.cuda import memcpy, blockIdx, threadIdx, shfl_down_sync, shfl_sync, blockDim
        from hidet.ir.primitives.math import exp, sqrt
        from hidet.ir.primitives import runtime

        _query, _seq_lengths, _cache_blocks, _key_cache, _value_cache = self.inputs
        bs, num_heads, _, head_size = self.inputs[0].shape
        max_cache_blocks = _cache_blocks.shape[-1]
        num_blocks, num_kv_heads, head_size, block_size = _key_cache.shape

        tile_size = 128

        assert int(32 % block_size) == 0
        with hidet.script_module() as script_module:

            @hidet.script
            def page_attention_score(
                max_seq_length: i32,
                score_ptr: ~f32,
                query: f16[bs, num_heads, head_size],
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = (max_seq_length + tile_size - 1) // tile_size, num_heads, bs
                attrs.cuda.block_dim = tile_size

                bs_idx = blockIdx.z
                head_idx = blockIdx.y

                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                j = blockIdx.x * tile_size + threadIdx.x
                seq_length = seq_lengths[bs_idx]

                if j < seq_length:
                    acc = f16.zero
                    block_idx = cache_blocks[bs_idx, j // block_size]
                    block_offset = j % block_size
                    kv_head_idx = head_idx % num_kv_heads
                    for k in range(head_size):
                        a = query[bs_idx, head_idx, k]
                        b = key_cache[block_idx, kv_head_idx, k, block_offset]
                        acc += a * b
                    acc = acc / sqrt(cast(head_size, f32))
                    score[bs_idx, head_idx, j] = acc

            @hidet.script
            def warp_max(val: f32) -> f32:
                attrs.func_kind = 'cuda_internal'
                for i in range(5):
                    val = max(val, shfl_down_sync(0xFFFFFFFF, val, 1 << i))
                val = shfl_sync(0xFFFFFFFF, val, 0)
                return val

            @hidet.script
            def warp_sum(val: f32) -> f32:
                attrs.func_kind = 'cuda_internal'
                for i in range(5):
                    val = val + shfl_down_sync(0xFFFFFFFF, val, 1 << i)
                val = shfl_sync(0xFFFFFFFF, val, 0)
                return val

            @hidet.script
            def page_attention_softmax(max_seq_length: i32, output_ptr: ~f32, score_ptr: ~f32, seq_lengths: i32[bs]):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = num_heads, bs
                attrs.cuda.block_dim = 32

                output = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=output_ptr)
                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                bs_idx = blockIdx.y
                head_idx = blockIdx.x

                seq_length = seq_lengths[bs_idx]
                warp_size = 32

                # max value
                max_val = f32.min_value
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        max_val = max(max_val, score[bs_idx, head_idx, j])
                max_val = warp_max(max_val)

                # sum exp
                sum_exp = f32.zero
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        sum_exp += exp(score[bs_idx, head_idx, j] - max_val)
                sum_exp = warp_sum(sum_exp)

                # divide
                for i in range((seq_length + warp_size - 1) / warp_size):
                    j = i * blockDim.x + threadIdx.x
                    if j < seq_length:
                        output[bs_idx, head_idx, j] = exp(score[bs_idx, head_idx, j] - max_val) / sum_exp

            @hidet.script
            def page_attention_output(
                max_seq_length: i32,
                output: f16[bs, num_heads, 1, head_size],
                score_ptr: ~f32,
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = num_heads, bs
                attrs.cuda.block_dim = head_size

                bs_idx = blockIdx.y
                head_idx = blockIdx.x
                kv_head_idx = head_idx % num_kv_heads

                score = tensor_pointer(f32, [bs, num_heads, max_seq_length], init=score_ptr)

                j = threadIdx.x
                seq_length = seq_lengths[bs_idx]

                acc = f32.zero

                for k in range(seq_length):
                    a = score[bs_idx, head_idx, k]
                    block_idx = cache_blocks[bs_idx, k // block_size]
                    block_offset = k % block_size
                    b = value_cache[block_idx, kv_head_idx, j, block_offset]
                    acc += a * b

                output[bs_idx, head_idx, 0, j] = cast(acc, f16)

            @hidet.script
            def launch(
                query: f16[bs, num_heads, 1, head_size],
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                key_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: f16[num_blocks, num_kv_heads, head_size, block_size],
                output: f16[bs, num_heads, 1, head_size],
            ):
                attrs.func_kind = 'public'

                # calculate max_seq_length
                max_seq_length: i32 = 0
                seq_lengths_cpu = cast(
                    runtime.request_cpu_workspace(nbytes=bs * i32.nbytes), dtype=tensor_pointer_type(i32, [bs])
                )
                memcpy(dst=seq_lengths_cpu, src=seq_lengths, count=bs * i32.nbytes, kind='cuda_to_cpu')
                for i in range(bs):
                    max_seq_length = max(max_seq_length, seq_lengths_cpu[i])

                # allocate cuda buffers
                cuda_workspace = cast(
                    runtime.request_cuda_workspace(nbytes=2 * bs * num_heads * max_seq_length * f32.nbytes), dtype=~u8
                )
                score = cast(~cuda_workspace[0], dtype=tensor_pointer_type(f32, [bs, num_heads, max_seq_length]))
                softmax = cast(
                    ~cuda_workspace[bs * num_heads * max_seq_length * f32.nbytes],
                    dtype=tensor_pointer_type(f32, [bs, num_heads, max_seq_length]),
                )

                # score = query @ key / sqrt(head_size)
                page_attention_score(max_seq_length, score, query, seq_lengths, cache_blocks, key_cache)

                # softmax(score)
                page_attention_softmax(max_seq_length, softmax, score, seq_lengths)

                # output = softmax @ value
                page_attention_output(max_seq_length, output, softmax, seq_lengths, cache_blocks, value_cache)

        return script_module.ir_module()


class PageAttentionOpV2(OpaqueOperator):
    def __init__(
        self,
        query: Tensor,
        seq_lengths: Tensor,
        cache_blocks: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        max_context_len: int,
    ):
        super().__init__(
            name='page_attentionV2',
            inputs={
                'query': query,
                'seq_lengths': seq_lengths,
                'cache_blocks': cache_blocks,
                'key_cache': key_cache,
                'value_cache': value_cache,
            },
            attributes={'max_context_len': max_context_len, 'qk_scale': query.shape[-1] ** -0.5},
        )

    def symbolic_forward(
        self, query: Tensor, seq_lengths: Tensor, cache_blocks: Tensor, key_cache: Tensor, value_cache: Tensor
    ):
        assert query.dtype == key_cache.dtype == value_cache.dtype, 'Mismatched dtype of query, key, value'
        assert query.dtype in [float16, float32], f'Unsupported dtype: {query.dtype}'
        bs, num_heads, _, head_size = query.shape
        return {'output': self.symbol(shape=[bs, num_heads, 1, head_size], dtype=query.dtype, device=query.device)}

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_cuda)

    @tune.space(1, num_threads=[128], partition_size=[0])  # empty space
    def schedule_cuda(self, num_threads: int = 128, partition_size: int = 0) -> IRModule:
        # naive implementation, todo: optimize this kernel
        import hidet
        from hidet.lang import attrs, cast, printf, shared_tensor
        from hidet.lang.types import u8, f16, f32, i32, i64, register_tensor, tensor_pointer_type, tensor_pointer
        from hidet.lang.types import int8, int16, int32, int64
        from hidet.lang.cuda import (
            memcpy,
            blockIdx,
            threadIdx,
            shfl_down_sync,
            shfl_xor_sync,
            shfl_sync,
            syncthreads,
            dynamic_shared_memory,
            blockDim,
            gridDim,
        )
        from hidet.ir.primitives.math import exp, sqrt
        from hidet.ir.primitives import runtime
        from hidet.ir.dtypes.vector import vectorize, float32x2, float32x4

        _query, _seq_lengths, _cache_blocks, _key_cache, _value_cache = self.inputs
        dtype = self.inputs[0].dtype

        qk_scale = self.attrs['qk_scale']
        bs, num_heads_, _, head_size = self.inputs[0].shape
        max_cache_blocks = _cache_blocks.shape[-1]
        num_blocks, num_kv_heads, head_size, block_size = _key_cache.shape

        WARP_SIZE = 32
        THREAD_GROUP_SIZE = max(WARP_SIZE // block_size, 1)
        VEC_SIZE = max(16 // (THREAD_GROUP_SIZE * dtype.nbytes), 1)
        V_VEC_SIZE = min(16 // dtype.nbytes, block_size)
        NUM_ELEMS_PER_THREAD = head_size // THREAD_GROUP_SIZE
        NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD // VEC_SIZE
        NUM_TOKENS_PER_THREAD_GROUP = (block_size + WARP_SIZE - 1) // WARP_SIZE
        NUM_WARPS = num_threads // WARP_SIZE

        padded_context_len = ((self.attrs['max_context_len'] + block_size - 1) // block_size) * block_size
        logits_size = padded_context_len * f32.nbytes
        outputs_size = (NUM_WARPS // 2) * head_size * dtype.nbytes
        shared_mem_size = max(logits_size, outputs_size)

        # only the bitwidths matter, this should be done more cleanly
        vec_table = {1: int8, 2: int16, 4: int32, 8: int64, 16: float32x4}
        vectorize_integral = lambda dtype, size: vec_table[dtype.nbytes * size]
        vec_dtype = vectorize_integral(dtype, VEC_SIZE)
        v_vec_dtype = vectorize_integral(dtype, V_VEC_SIZE)
        if f32.nbytes * V_VEC_SIZE > 16:
            v_vec_logit_dtype = float32x4
            v_vec_logit_loads = f32.nbytes * V_VEC_SIZE // 16
        else:
            v_vec_logit_dtype = vectorize_integral(f32, V_VEC_SIZE)
            v_vec_logit_loads = 1

        assert int(32 % block_size) == 0
        with hidet.script_module() as script_module:

            @hidet.script
            def qk_dot(q: ~vec_dtype, k: ~vec_dtype) -> f32:
                attrs.func_kind = 'cuda_internal'
                accum = f32.zero
                for i in range(NUM_VECS_PER_THREAD):
                    q0 = q[i]
                    k0 = k[i]
                    for j in range(VEC_SIZE):
                        accum += cast(~q0, ~dtype)[j] * cast(~k0, ~dtype)[j]

                # sum across lanes
                m = THREAD_GROUP_SIZE // 2
                while m >= 1:
                    accum += shfl_xor_sync(0xFFFFFFFF, accum, m)
                    m //= 2

                return accum

            @hidet.script
            def block_sum(smem: ~f32, val: f32) -> f32:
                attrs.func_kind = 'cuda_internal'
                warp = threadIdx.x // WARP_SIZE
                lane = threadIdx.x % WARP_SIZE

                m = WARP_SIZE // 2
                while m >= 1:
                    val += shfl_xor_sync(0xFFFFFFFF, val, m)
                    m //= 2

                if lane == 0:
                    smem[warp] = val

                syncthreads()

                val = smem[lane] if lane < NUM_WARPS else f32.zero

                m = NUM_WARPS // 2
                while m >= 1:
                    val += shfl_xor_sync(0xFFFFFFFF, val, m)
                    m //= 2

                return shfl_sync(0xFFFFFFFF, val, 0)

            @hidet.script
            def cdiv(a: i32, b: i32) -> i32:
                attrs.func_kind = 'cuda_internal'
                return (a + b - 1) // b

            @hidet.script
            def page_attention_kernel(
                exp_sums: ~f32,
                max_logits: ~f32,
                out: ~dtype,
                q: ~dtype,
                k_cache: ~dtype,
                v_cache: ~dtype,
                num_kv_heads: i32,
                scale: f32,
                block_tables: ~i32,
                context_lens: ~i32,
                max_num_blocks_per_seq: i32,
                q_stride: i32,
                kv_block_stride: i32,
                kv_head_stride: i32,
            ):
                attrs.func_kind = 'cuda_kernel'
                attrs.cuda.grid_dim = (num_heads_, bs)
                attrs.cuda.block_dim = num_threads
                attrs.cuda.dynamic_smem_bytes = shared_mem_size

                seq_idx = blockIdx.y
                partition_idx = blockIdx.z
                max_num_partitions = gridDim.z
                context_len = context_lens[seq_idx]

                USE_PARTITIONING = partition_size > 0
                if USE_PARTITIONING and partition_idx * partition_size >= context_len:
                    return

                num_context_blocks = (context_len + block_size - 1) // block_size
                num_blocks_per_partition = num_context_blocks
                if USE_PARTITIONING:
                    num_blocks_per_partition = partition_size // block_size

                start_block_idx = 0
                if USE_PARTITIONING:
                    start_block_idx = partition_idx * num_blocks_per_partition
                end_block_idx = min(start_block_idx + num_blocks_per_partition, num_context_blocks)
                num_blocks = end_block_idx - start_block_idx

                start_token_idx = start_block_idx * block_size
                end_token_idx = min(start_token_idx + num_blocks * block_size, context_len)
                num_tokens = end_token_idx - start_token_idx

                NUM_THREAD_GROUPS = num_threads // THREAD_GROUP_SIZE
                assert num_threads % THREAD_GROUP_SIZE == 0

                thread_idx = threadIdx.x
                warp_idx = thread_idx // WARP_SIZE
                lane = thread_idx % WARP_SIZE

                head_idx = blockIdx.x
                num_heads = gridDim.x
                num_queries_per_kv = num_heads // num_kv_heads
                kv_head_idx = head_idx // num_queries_per_kv

                thread_group_idx = thread_idx // THREAD_GROUP_SIZE
                thread_group_offset = thread_idx % THREAD_GROUP_SIZE

                q_ptr = q + seq_idx * q_stride + head_idx * head_size
                q_vecs = shared_tensor(vec_dtype, shape=[THREAD_GROUP_SIZE, NUM_VECS_PER_THREAD])

                # for i in range(thread_group_idx, NUM_VECS_PER_THREAD, NUM_THREAD_GROUPS):
                i1 = thread_group_idx
                while i1 < NUM_VECS_PER_THREAD:
                    vec_idx = thread_group_offset + i1 * THREAD_GROUP_SIZE
                    q_ptr_1 = q_ptr + vec_idx * VEC_SIZE
                    q_vecs[thread_group_offset][i1] = cast(q_ptr_1, ~vec_dtype)[0]
                    i1 += NUM_THREAD_GROUPS

                syncthreads()
                logits = dynamic_shared_memory(0, f32)
                red_smem = shared_tensor(f32, shape=[2 * NUM_WARPS])

                x = 16 // dtype.nbytes
                qk_max = -f32.max_value

                block_table = block_tables + seq_idx * max_num_blocks_per_seq
                # for block_idx in range(start_block_idx + warp_idx, end_block_idx, NUM_WARPS):
                block_idx = start_block_idx + warp_idx
                while block_idx < end_block_idx:
                    physical_block_number = int64(0)
                    physical_block_number = block_table[block_idx]

                    for i in range(NUM_TOKENS_PER_THREAD_GROUP):
                        physical_block_offset = (thread_group_idx + i * WARP_SIZE) % block_size
                        token_idx = block_idx * block_size + physical_block_offset
                        k_vecs = register_tensor(vec_dtype, shape=[NUM_VECS_PER_THREAD])

                        for j in range(NUM_VECS_PER_THREAD):
                            k_ptr = (
                                k_cache
                                + physical_block_number * kv_block_stride
                                + kv_head_idx * kv_head_stride
                                + physical_block_offset * x
                            )
                            vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE
                            offset1 = (vec_idx * VEC_SIZE) // x
                            offset2 = (vec_idx * VEC_SIZE) % x
                            k_ptr_1 = k_ptr + offset1 * block_size * x + offset2
                            k_vecs[j] = cast(k_ptr_1, ~vec_dtype)[0]

                        qk = scale * qk_dot(~q_vecs[thread_group_offset][0], ~k_vecs[0])

                        if thread_group_offset == 0:
                            mask = token_idx >= context_len
                            logits[token_idx - start_token_idx] = 0.0 if mask else qk
                            qk_max = qk_max if mask else max(qk_max, qk)

                    block_idx += NUM_WARPS

                m1 = WARP_SIZE // 2
                while m1 >= THREAD_GROUP_SIZE:
                    qk_max = max(qk_max, shfl_xor_sync(0xFFFFFFFF, qk_max, m1))
                    m1 //= 2
                if lane == 0:
                    red_smem[warp_idx] = qk_max
                syncthreads()

                qk_max = red_smem[lane] if lane < NUM_WARPS else -f32.max_value
                m1 = NUM_WARPS // 2
                while m1 >= 1:
                    qk_max = max(qk_max, shfl_xor_sync(0xFFFFFFFF, qk_max, m1))
                    m1 //= 2
                syncthreads()
                qk_max = shfl_sync(0xFFFFFFFF, qk_max, 0)

                exp_sum = f32.zero
                # for i in range(thread_idx, num_tokens, num_threads):
                i2 = thread_idx
                while i2 < num_tokens:
                    val = exp(logits[i2] - qk_max)
                    logits[i2] = val
                    exp_sum += val
                    i2 += num_threads

                exp_sum = block_sum(~red_smem[NUM_WARPS], exp_sum)

                inv_sum = 1 / (exp_sum + 1e-6)

                # for i in range(thread_idx, num_tokens, num_threads):
                i3 = thread_idx
                while i3 < num_tokens:
                    logits[i3] *= inv_sum
                    i3 += num_threads

                syncthreads()

                if USE_PARTITIONING and thread_idx == 0:
                    max_logits_ptr = (
                        max_logits
                        + seq_idx * num_heads * max_num_partitions
                        + head_idx * max_num_partitions
                        + partition_idx
                    )
                    max_logits_ptr[0] = qk_max
                    exp_sums_ptr = (
                        exp_sums
                        + seq_idx * num_heads * max_num_partitions
                        + head_idx * max_num_partitions
                        + partition_idx
                    )
                    exp_sums_ptr[0] = exp_sum

                NUM_V_VECS_PER_ROW = block_size // V_VEC_SIZE
                NUM_ROWS_PER_ITER = WARP_SIZE // NUM_V_VECS_PER_ROW
                NUM_ROWS_PER_THREAD = (head_size + NUM_ROWS_PER_ITER - 1) // NUM_ROWS_PER_ITER

                accs = register_tensor(f32, [NUM_ROWS_PER_THREAD])

                for i in range(NUM_ROWS_PER_THREAD):
                    accs[i] = 0.0

                # for block_idx in range(start_block_idx + warp_idx, end_block_idx, NUM_WARPS):
                block_idx = start_block_idx + warp_idx
                while block_idx < end_block_idx:
                    physical_block_number = cast(block_table[block_idx], i64)
                    physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE
                    token_idx = block_idx * block_size + physical_block_offset
                    logits_vec = register_tensor(v_vec_logit_dtype, [v_vec_logit_loads])
                    for i in range(v_vec_logit_loads):
                        logits_vec[i] = cast(logits + token_idx - start_token_idx, ~v_vec_logit_dtype)[i]

                    v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride

                    for i in range(NUM_ROWS_PER_THREAD):
                        row_idx = lane // NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER
                        if row_idx < head_size:
                            offset = row_idx * block_size + physical_block_offset
                            v_vec = cast(v_ptr + offset, ~v_vec_dtype)[0]
                            if block_idx == num_context_blocks - 1:
                                for j in range(V_VEC_SIZE):
                                    if token_idx + j < context_len:
                                        cast(~v_vec, ~dtype)[j] = cast(~v_vec, ~dtype)[j]
                                    else:
                                        cast(~v_vec, ~dtype)[j] = dtype(0.0)

                            for j in range(V_VEC_SIZE):
                                lv = cast(~logits_vec, ~f32)[j]
                                rv = cast(cast(~v_vec, ~dtype)[j], f32)
                                k = lv * rv
                                accs[i] += k

                    block_idx += NUM_WARPS

                for i in range(NUM_ROWS_PER_THREAD):
                    acc = accs[i]
                    m1 = NUM_V_VECS_PER_ROW // 2
                    while m1 >= 1:
                        acc += shfl_xor_sync(0xFFFFFFFF, acc, m1)
                        m1 //= 2
                    accs[i] = acc

                syncthreads()

                out_smem = cast(logits, ~f32)

                m1 = NUM_WARPS
                while m1 > 1:
                    mid = m1 // 2

                    if warp_idx >= mid and warp_idx < m1:
                        dst = out_smem + (warp_idx - mid) * head_size

                        for j in range(NUM_ROWS_PER_THREAD):
                            row_idx = lane // NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER
                            if row_idx < head_size and lane % NUM_V_VECS_PER_ROW == 0:
                                dst[row_idx] = accs[j]

                    syncthreads()

                    if warp_idx < mid:
                        src = out_smem + warp_idx * head_size
                        for j in range(NUM_ROWS_PER_THREAD):
                            row_idx = lane // NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER
                            if row_idx < head_size and lane % NUM_V_VECS_PER_ROW == 0:
                                accs[j] += src[row_idx]

                    syncthreads()

                    m1 //= 2

                if warp_idx == 0:
                    out_ptr = (
                        out
                        + seq_idx * max_num_partitions * num_heads * head_size
                        + head_idx * max_num_partitions * head_size
                        + partition_idx * head_size
                    )

                    for i in range(NUM_ROWS_PER_THREAD):
                        row_idx = lane // NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER
                        if row_idx < head_size and lane % NUM_V_VECS_PER_ROW == 0:
                            out_ptr[row_idx] = cast(accs[i], dtype)

            @hidet.script
            def launch(
                query: dtype[bs, num_heads_, 1, head_size],
                seq_lengths: i32[bs],
                cache_blocks: i32[bs, max_cache_blocks],
                key_cache: dtype[num_blocks, num_kv_heads, head_size, block_size],
                value_cache: dtype[num_blocks, num_kv_heads, head_size, block_size],
                output: dtype[bs, num_heads_, 1, head_size],
            ):
                attrs.func_kind = 'public'

                page_attention_kernel(
                    0,
                    0,  # exp_sums, max_logits
                    output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    qk_scale,
                    cache_blocks,
                    seq_lengths,
                    max_cache_blocks,
                    num_heads_ * head_size,  # q_stride
                    num_kv_heads * head_size * block_size,  # kv_block_stride
                    head_size * block_size,  # kv_head_stride
                )

        return script_module.ir_module()


def cache_write(
    seq_lengths: Tensor, key: Tensor, value: Tensor, cache_slots: Tensor, key_cache: Tensor, value_cache: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Write the key and value to the cache.

    Parameters
    ----------
    seq_lengths: Tensor
        The sequence lengths. Shape: i32 [bs]
    key: Tensor
        The key tensor. Shape: [bs, num_kv_heads, max_seq_length, head_size]
    value: Tensor
        The value tensor. Shape: [bs, num_kv_heads, max_seq_length, head_size]
    cache_slots: Tensor
        The cache slots. Shape: i64 [bs, max_seq_length]
    key_cache: Tensor
        The key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: Tensor
        The value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]

    Returns
    -------
    (updated_key_cache, updated_value_cache): Tuple[Tensor, Tensor]
        updated_key_cache: The updated key cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
        updated_value_cache: The updated value cache. Shape: [num_blocks, num_kv_heads, head_size, block_size]
    """
    return PageAttentionWriteCacheOp(seq_lengths, key, value, cache_slots, key_cache, value_cache).outputs


def page_attention(
    query: Tensor,
    seq_lengths: Tensor,
    cache_blocks: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    max_context_len: int = 1024,
):
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
    return PageAttentionOpV2(query, seq_lengths, cache_blocks, key_cache, value_cache, max_context_len).outputs[0]
