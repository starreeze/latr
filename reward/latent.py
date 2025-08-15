from typing import Any

from args import DataArgs, RewardArgs
from model.meta.base import TokenConstants
from reward.base import RewardBase
from tools.utils import count_latent


class AnswerLenReward(RewardBase):
    """
    length of the explicit answer should fall in the range of
    [min_full_score_len, max_full_score_len] to get full score.
    if non think model, the length should be less than nothink_max_len.
    if the length is smaller than min_full_score_len
    or larger than max_full_score_len, the score will be linearly decreasing.
    """

    min_full_score_len = 30
    max_full_score_len = 1000
    max_no_score_len = 2000
    nothink_max_len = 30

    def _calc_score(self, completion: str, answer: Any) -> float:
        end_marker_pos = completion.find(self.end_marker)
        if end_marker_pos == -1:
            return 0
        answer_len = len(completion[end_marker_pos:].split())
        if not self.reasoning_model:
            score = int(answer_len < self.nothink_max_len)
        elif answer_len < self.min_full_score_len:
            score = answer_len / 100
        elif answer_len > self.max_full_score_len:
            effective_len = min(answer_len, self.max_no_score_len)
            score = 1 - (effective_len - self.max_full_score_len) / (
                self.max_no_score_len - self.max_full_score_len
            )
        else:
            score = 1
        return self.reward_args.answer_len_reward * score


class AnswerNoThinkReward(RewardBase):
    """
    if the answer contains think words, the score will be linearly decreasing.
    """

    think_words = ["hmm", "wait,", "okay,", "i need"]
    max_no_score_think_words = 5

    def _calc_score(self, completion: str, answer: Any) -> float:
        extracted_answer = self.extractor(completion).lower()
        if not extracted_answer:
            return 0
        think_count = sum(extracted_answer.count(word) for word in self.think_words)
        score = max(0, 1 - think_count / self.max_no_score_think_words)
        return self.reward_args.answer_no_think_reward * score


class ResponseThinkReward(RewardBase):
    """
    there should be at least n tokens / words to think;
    linearly increase the score from 0 to 1 if the think length is less than n tokens.
    """

    min_think_len = 500

    def _calc_score(self, completion: str, answer: Any) -> float:
        # we need to manually determine the first think block since sometimes
        # the think start marker is in the prompt
        start_pos = completion.find(self.start_marker)
        end_pos = completion.find(self.end_marker)
        if start_pos == -1 or end_pos == -1:
            return 0
        if end_pos < start_pos:
            # probably the think start marker is in the prompt
            start_pos = 0
        else:
            start_pos += len(self.start_marker)
        think_block = completion[start_pos:end_pos]
        num_latent = count_latent(TokenConstants.latent_tokens[2], think_block)
        if num_latent == 0:
            num_latent = len(think_block.split())
        score = min(num_latent / self.min_think_len, 1)
        return self.reward_args.response_think_reward * score


class LatentLenReward(RewardBase):
    max_correct_len = 0.5
    min_wrong_len = 0.9

    def __init__(
        self,
        correctness_reward_fn: RewardBase,
        latent_token: str,
        max_latent_len: int,
        reward_args: RewardArgs,
        data_args: DataArgs,
        reasoning_model: bool,
    ):
        super().__init__(reward_args, data_args, reasoning_model)
        self.correctness_reward_fn = correctness_reward_fn
        self.latent_token = latent_token
        self.max_latent_len = max_latent_len

    def _calc_score(self, completion: str, answer: Any) -> float:
        ratio = count_latent(self.latent_token, completion) / self.max_latent_len
        if self.correctness_reward_fn._calc_score(completion, answer):
            score = 1 - max((ratio - self.max_correct_len) / (1 - self.max_correct_len), 0)
        else:
            score = min((ratio - self.min_wrong_len) / self.min_wrong_len, 0)
        return self.reward_args.latent_len_reward * score
