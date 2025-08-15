from abc import ABC, abstractmethod
from typing import Any

from args import DataArgs, RewardArgs
from data.eval import AnswerExtractor


class RewardBase(ABC):
    def __init__(self, reward_args: RewardArgs, data_args: DataArgs, reasoning_model: bool):
        self.reward_args = reward_args
        self.extractor = AnswerExtractor(data_args)
        self.__name__ = self.__class__.__name__
        self.start_marker = data_args.plain_cot_start_marker
        self.end_marker = data_args.plain_cot_end_marker
        self.reasoning_model = reasoning_model
        self.template = data_args.template

    @abstractmethod
    def _calc_score(self, completion: str, answer: str | dict) -> float:
        pass

    def __call__(
        self,
        completions: list[list[dict[str, Any]]] | list[str],
        answer: list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[float]:
        assert len(completions) == len(
            answer
        ), f"The number of completions ({len(completions)}) and answer ({len(answer)}) must be the same."
        if self.template == "conv":
            assert isinstance(completions[0], list)
            contents = [completion[0]["content"] for completion in completions]  # type: ignore
        else:
            assert isinstance(completions[0], str)
            contents = completions
        return [self._calc_score(c, a) for c, a in zip(contents, answer)]

    def __str__(self) -> str:
        return self.__name__

    def __repr__(self) -> str:
        return self.__name__
