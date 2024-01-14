from typing import List, Optional
from enum import Enum
from .sampler import SamplingParams, SamplerOutput
from .cache import CacheTableManager, BlockDevice


class SequenceState(Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED_STOPPED = 'finished_stopped'
    FINISHED_LENGTH = 'finished_length'


class Sequence:
    def __init__(self, sequence_id: int, prompt: str, sampling_params: SamplingParams):
        self.sequence_id: int = sequence_id
        self.prompt: str = prompt
        self.sampling_params: SamplingParams = sampling_params

        self.prompt_tokens: List[int] = []
        self.output_tokens: List[int] = []
        self.blocks: List[int] = []
        self.status: SequenceState = SequenceState.WAITING

    def append_token(self, token: int):
        self.output_tokens.append(token)

        # update status
        if token in self.sampling_params.stop_token_ids:
            self.status = SequenceState.FINISHED_STOPPED
        elif len(self.output_tokens) >= self.sampling_params.max_tokens:
            self.status = SequenceState.FINISHED_LENGTH
        else:
            self.status = SequenceState.RUNNING

    def is_finished(self) -> bool:
        return self.status in [SequenceState.FINISHED_STOPPED, SequenceState.FINISHED_LENGTH]


class SequenceOutput:
    def __init__(
        self,
        sequence_id: int,
        prompt: str,
        output_text: str,
        prompt_tokens: List[int],
        output_tokens: List[int],
        status: SequenceState,
    ):
        self.sequence_id: int = sequence_id
        self.prompt: str = prompt
        self.output_text: str = output_text
        self.prompt_tokens: List[int] = prompt_tokens
        self.output_tokens: List[int] = output_tokens
        self.status: SequenceState = status

    def is_finished(self) -> bool:
        return self.status in [SequenceState.FINISHED_STOPPED, SequenceState.FINISHED_LENGTH]


class SequenceScheduler:
    def __init__(self, cache: CacheTableManager):
        self.cache: CacheTableManager = cache
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

        # allocate virtual blocks for the sequence
        num_blocks: int = (len(sequence.prompt_tokens) + self.cache.block_size - 1) // self.cache.block_size
        sequence.blocks.extend(self.cache.alloc_virtual_blocks(num_blocks))

    def schedule(self) -> List[Sequence]:
        # current strategy: put all waiting requests into running list, and raise an error if there is not enough blocks
        # todo: implement swapping strategy
        while len(self.waiting) > 0:
            seq = self.waiting.pop()

            # all virtual blocks are not mapped to physical blocks yet, allocate gpu blocks for them
            gpu_blocks: List[int] = self.cache.alloc_gpu_blocks(len(seq.blocks))

            # map virtual blocks to gpu blocks
            for vir_block, gpu_block in zip(seq.blocks, gpu_blocks):
                self.cache.map_block(vir_block, BlockDevice.GPU, gpu_block)

            # add the sequence to running list
            self.running.append(seq)

        return self.running

    def update(self):
        self.running = [sequence for sequence in self.running if not sequence.is_finished()]
