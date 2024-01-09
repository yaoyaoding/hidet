from typing import List
from .sample import SamplingParams


class Sequence:
    def __init__(self, prompt: str, prompt_tokens: List[int], sampling_params: SamplingParams):
        self.prompt: str = prompt
        self.prompt_tokens: List[int] = prompt_tokens
        self.sampling_params: SamplingParams = sampling_params

        self.output_tokens: List[int] = []
        self.blocks: List[int] = []


class SequenceScheduler:
    def __init__(self):
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def step(self) -> List[Sequence]:
        pass
