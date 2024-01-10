"""

The LLM app is an application built on top of the hidet compiler, used to perform the completion task of the LLM model.

Inside the LLM app, there are two main computation graphs: prefill and decode.

The prefill graph takes the following inputs and outputs:
    inputs:
        input_ids: int32 [bs, seq_len]
        position_ids: int32 [bs, seq_len]
        cache_slots: int64 [bs, seq_len]
        seq_lengths: int32 [bs]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]   (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    outputs:
        hidden_states: dtype [bs, seq_len, hidden_size]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    (Note: the key_caches in the inputs and outputs share the same memory, similarly for value_caches)

The decode graph takes the following inputs and outputs:
    inputs:
        input_ids: int32 [bs, 1]
        position_ids: int32 [bs, 1]
        cache_slots: int64 [bs, 1]
        seq_lengths: int32 [bs]
        max_context_length: int32
        cache_blocks: int32 [bs, max_num_cache_blocks]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]   (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    outputs:
        hidden_states: dtype [bs, 1, hidden_size]
        *key_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
        *value_caches: dtype [num_blocks, num_heads, head_size, block_size]  (num_layers)
    (Note: the key_caches in the inputs and outputs share the same memory, similarly for value_caches)


The LLM app supports two operations:
1. add a sequence to the app
2. perform a step of scheduling and running, which will select a batch of sequences to run and return the outputs
   of the selected sequences.

Acknowledgement:
    - We adopt the page attention mechanism proposed in vLLM: https://github.com/vllm-project/vllm
"""
from typing import List, Tuple, Optional
import dataclasses
import torch
from hidet.runtime.compiled_app import CompiledApp, AppMetaData
from hidet.ir.type import data_type
from hidet.graph.tensor import Tensor
from hidet.apps.llm.sampler import SamplingParams
from hidet.apps.llm.sequence import Sequence, SequenceScheduler, SequenceOutput
from hidet.apps.llm.cache import CacheTable
from hidet.apps.llm.sampler import Sampler, SamplerOutput
from hidet.apps.llm.tokenizer import Tokenizer
from hidet.utils.dataclass import from_dict
from .utils import tensor_pad

@dataclasses.dataclass
class Attributes:
    cache_dtype: str
    num_layers: int
    num_heads: int
    head_size: int
    block_size: int
    tokenizer: str  # currently, we use the tokenizer from huggingface


class LLM:
    def __init__(self, compiled_app: CompiledApp, memory_capacity: Optional[int] = None):
        super().__init__()
        self.compiled_app: CompiledApp = compiled_app
        self.attributes: Attributes = from_dict(Attributes, compiled_app.attributes)
        self.scheduler: SequenceScheduler = SequenceScheduler()
        self.sampler: Sampler = Sampler(embedding=self.compiled_app.tensors['embedding'])
        self.tokenizer: Tokenizer = Tokenizer(self.attributes.tokenizer)
        self.cache: CacheTable = CacheTable(
            dtype=data_type(self.attributes.cache_dtype),
            memory_capacity=memory_capacity,
            num_layers=self.attributes.num_layers,
            num_heads=self.attributes.num_heads,
            head_size=self.attributes.head_size,
            block_size=self.attributes.block_size,
        )

    def _prefill_forward(
        self,
        input_ids: Tensor,  # int32 [bs, seq_len]
        position_ids: Tensor,  # int32 [bs, seq_len]
        cache_slots: Tensor,  # int64 [bs, seq_len]
        seq_lengths: Tensor  # int32 [bs]
    ) -> Tensor:
        # prefill compiled graph:
        pass

    def _decode_forward(
        self,
        input_ids: Tensor,  # int32 [bs, 1]
        position_ids: Tensor,  # int32 [bs, 1]
        cache_slots: Tensor,  # int64 [bs, 1]
        seq_lengths: Tensor,  # int32 [bs]
        max_context_length: int,  # int32
        cache_blocks: Tensor,  # int32 [bs, max_num_cache_blocks]
    ) -> Tensor:
        pass

    def _prefill(self, sequences: List[Sequence]) -> Tensor:
        import hidet

        max_length = max(len(seq.prompt_tokens) for seq in sequences)
        input_ids: Tensor = tensor_pad([seq.prompt_tokens for seq in sequences], max_length)
        position_ids: Tensor = tensor_pad([list(range(len(seq.prompt_tokens))) for seq in sequences], max_length)
        # cache_slots: Tensor =
        #
        # hidden_states: Tensor = self._prefill_forward(*inputs)  # [bs, seq_len, hidden_size]


    def _decode(self, sequences: List[Sequence]) -> Tensor:
        inputs: List[Tensor] = self._prepare_decode_inputs(sequences)
        hidden_states: Tensor = self._decode_forward(*inputs)  # [bs, seq_len, hidden_size]

        # sample the next token given the hidden states
        sampler_outputs: List[SamplerOutput] = self.sampler.sample(sequences, hidden_states)
        return sampler_outputs

    def _post_process(self, sampler_outputs: List[SamplerOutput]) -> List[SequenceOutput]:
        for sequence, sequence_output in zip(self.scheduler.running, sampler_outputs):
            sequence.append_token(sequence_output.token)
            if sequence.is_finished():
                sequence_output.text = self.tokenizer.decode(sequence.output_tokens)

    def add_sequence(self, sequence_id: int, prompt: str, sampling_params: SamplingParams):
        self.scheduler.waiting.append(
            Sequence(sequence_id, prompt, sampling_params)
        )

    def step(self) -> List[SequenceOutput]:
        # schedule for the next step, got the sequences to run
        sequences: List[Sequence] = self.scheduler.schedule()

        # run the sequences and get the next token for each sequence
        if all(len(seq.output_tokens) == 0 for seq in sequences):
            # prefill
            hidden_states = self._prefill(sequences)
        elif all(len(seq.output_tokens) > 0 for seq in sequences):
            # decode
            hidden_states = self._decode(sequences)
        else:
            raise ValueError("Some sequences are prefilling and some are decoding.")

        # sample the next token given the hidden states
        sampler_outputs: List[SamplerOutput] = self.sampler.sample(sequences, hidden_states)

        # append the next token for each sequence, incrementally detokenize the output text
        sequence_outputs: List[SequenceOutput] = self._post_process(sampler_outputs)

        # update the scheduler status (e.g., some sequences may be finished)
        self.scheduler.update()

        return sequence_outputs
