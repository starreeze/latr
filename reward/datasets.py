from typing import Any

from args import DataArgs, RewardArgs
from data.eval import (
    AnswerExtractor,
    CountdownAnswerMatcher,
    IdentityAnswerMatcher,
    validate_countdown_equation,
)
from reward.base import RewardBase


class CountdownCorrectnessReward(RewardBase):
    def __init__(self, reward_args: RewardArgs, data_args: DataArgs, reasoning_model: bool):
        super().__init__(reward_args, data_args, reasoning_model)
        self.extractor = AnswerExtractor(data_args)
        self.matcher = CountdownAnswerMatcher()

    def _calc_score(self, response: str, answer: dict[str, Any]) -> float:
        pred = self.extractor(response)
        return self.reward_args.correctness_reward * self.matcher(pred, answer)


class CountdownAnswerFormatReward(RewardBase):
    def __init__(self, reward_args: RewardArgs, data_args: DataArgs, reasoning_model: bool):
        super().__init__(reward_args, data_args, reasoning_model)
        self.extractor = AnswerExtractor(data_args)

    def _calc_score(self, response: str, answer: dict[str, Any]) -> float:
        pred = self.extractor(response)
        return self.reward_args.answer_format_reward * validate_countdown_equation(pred, answer["numbers"])


class IdentityCorrectnessReward(RewardBase):
    def __init__(self, reward_args: RewardArgs, data_args: DataArgs, reasoning_model: bool):
        super().__init__(reward_args, data_args, reasoning_model)
        self.extractor = AnswerExtractor(data_args)
        self.matcher = IdentityAnswerMatcher()

    def _calc_score(self, response: str, answer: str) -> float:
        pred = self.extractor(response)
        return self.reward_args.correctness_reward * self.matcher(pred, {"answer": answer})


class DigitAnswerFormatReward(RewardBase):
    def __init__(self, reward_args: RewardArgs, data_args: DataArgs, reasoning_model: bool):
        super().__init__(reward_args, data_args, reasoning_model)
        self.extractor = AnswerExtractor(data_args)

    def _calc_score(self, response: str, answer: str) -> float:
        pred = self.extractor(response)
        return self.reward_args.answer_format_reward * pred.isdigit()


AimeCorrectnessReward = Gsm8kCorrectnessReward = DapomathCorrectnessReward = IdentityCorrectnessReward
AimeAnswerFormatReward = Gsm8kAnswerFormatReward = DapomathAnswerFormatReward = DigitAnswerFormatReward
