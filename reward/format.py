import re
from typing import Any

from reward.base import RewardBase

# usually start marker is enforced in generation in plain cot, so no need to check for it
plain_pattern = r"^(START_MARKER)?(.*?)END_MARKER(.*?)\\boxed\{(.*?)\}(.*?)$"


class PlainLooseFormatReward(RewardBase):
    def _calc_score(self, completion: str, answer: Any) -> float:
        return self.reward_args.loose_format_reward * all(
            m in completion for m in [self.end_marker, r"\boxed"]
        )


class PlainStrictFormatReward(RewardBase):
    def _calc_score(self, completion: str, answer: Any) -> float:
        pattern = plain_pattern.replace("END_MARKER", re.escape(self.end_marker)).replace(
            "START_MARKER", re.escape(self.start_marker)
        )
        match = re.match(pattern, completion, re.DOTALL)
        if match is None:
            return 0
        for g in match.groups()[1:]:  # the first is potential start_marker
            if any(m in g for m in [self.start_marker, self.end_marker, r"\boxed"]):
                return 0
        return self.reward_args.strict_format_reward


xml_pattern = r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*"


class XmlLooseFormatReward(RewardBase):
    no_reward_trailing_len = 100
    num_tags = 4

    def _calc_score(self, completion: str, answer: Any) -> float:
        score = 0.0
        tag_reward = 1 / self.num_tags
        penalty = tag_reward / self.no_reward_trailing_len

        if completion.count("<think>") == 1:
            score += tag_reward
        if completion.count("</think>") == 1:
            score += tag_reward
        if completion.count("<answer>") == 1:
            score += tag_reward
        if completion.count("</answer>") == 1:
            # Penalize trailing characters after </answer>
            r = tag_reward - len(completion.split("</answer>")[-1]) * penalty
            score += max(0.0, r)
        return score * self.reward_args.loose_format_reward


class XmlStrictFormatReward(RewardBase):
    def _calc_score(self, completion: str, answer: Any) -> float:
        return self.reward_args.strict_format_reward * (
            re.match(xml_pattern, completion, re.DOTALL) is not None
        )
