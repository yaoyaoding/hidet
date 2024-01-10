from typing import List, Optional
from enum import StrEnum
from .sampler import SamplingParams, SamplerOutput


class SequenceState(StrEnum):
    WAITING = 'waiting'
    RUNNING = 'running'
    FINISHED_STOPPED = 'finished_STOPPED'
    FINISHED_LENGTH = 'finished_LENGTH'


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
    def __init__(self):
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def schedule(self) -> List[Sequence]:
        pass

    def update(self):
        self.running = [sequence for sequence in self.running if not sequence.is_finished()]
