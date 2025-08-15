import re
from abc import ABC, abstractmethod
from typing import Any

from data.utils import (
    convert_countdown_answer,
    eval_countdown_equation,
    math_equal,
    simplify_math_expr,
    validate_countdown_equation,
)

# from args import DataArgs
DataArgs = Any


class AnswerExtractor:
    formula_pattern = r"([\d\+\-\*/=\(\)\.\,\s]+)"
    patterns = {
        "none": [r"</latent_think>(.*)$", formula_pattern],
        "plain": ["BOXED_PATTERN", formula_pattern],
        "xml": [r"<answer>(.*?)</answer>", formula_pattern],
    }

    def __init__(self, data_args: DataArgs):
        self.pattern_strs = self.patterns[data_args.cot_type]
        self.start_marker = data_args.plain_cot_start_marker if data_args.cot_type == "plain" else ""
        self.end_marker = data_args.plain_cot_end_marker if data_args.cot_type == "plain" else ""

    @staticmethod
    def get_boxed(response: str) -> str:
        if "boxed" not in response:
            return ""

        ans = response.rsplit("boxed", 1)[-1]
        if len(ans) == 0:
            return ""
        if ans[0] != "{":
            return ans.split("$")[0].strip()

        stack, i = 1, 1
        while i < len(ans):
            if ans[i] == "{":
                stack += 1
            elif ans[i] == "}":
                stack -= 1
                if stack == 0:
                    break
            i += 1
        return ans[1:i]

    def get_next_think_block(self, response: str, start: int) -> tuple[int, int]:
        """
        Find the (start, end) indices (exclusive) of the next matched block starting from 'start'.
        Returns (-1, -1) if no such block is found.
        The returned indices include the markers.
        """
        start_pos = response.find(self.start_marker, start)
        if start_pos == -1:
            return -1, -1

        stack = 1
        j = start_pos + len(self.start_marker)

        while j < len(response) and stack > 0:
            next_start = response.find(self.start_marker, j)
            next_end = response.find(self.end_marker, j)

            if next_end == -1:
                # No more end markers, treat rest as unmatched
                return start_pos, len(response)

            if next_start != -1 and next_start < next_end:
                stack += 1
                j = next_start + len(self.start_marker)
            else:
                stack -= 1
                j = next_end + len(self.end_marker)

        if stack == 0:
            return start_pos, j
        else:
            return -1, -1

    def remove_think_blocks(self, response: str) -> str:
        """
        Remove all blocks between the start_marker and end_marker (include the markers).
        """
        assert bool(self.start_marker) == bool(self.end_marker)
        if not self.start_marker:
            return response

        if self.start_marker == self.end_marker:
            # remove all blocks between the even-occurred marker and the odd-occurred marker
            marker = self.start_marker
            result = []
            parts = response.split(marker)

            # Keep the first part (before any marker)
            result.append(parts[0])

            # For subsequent parts, keep only the even-indexed ones (1-indexed counting)
            # This means we skip parts[1], parts[3], parts[5], etc. (the content between markers)
            # and keep parts[2], parts[4], parts[6], etc. (the content after closing markers)
            for i in range(2, len(parts), 2):
                result.append(parts[i])

            return "".join(result)
        else:
            # remove all blocks between the matched start_marker and end_marker using get_next_think_block
            result = []
            i = 0

            while i < len(response):
                start_pos, end_pos = self.get_next_think_block(response, i)
                if start_pos == -1:
                    result.append(response[i:])
                    break
                result.append(response[i:start_pos])
                i = end_pos

            return "".join(result)

    def __call__(self, response: str) -> str:
        # remove think blocks to ensure a fair comparison between latent and explicit
        response = self.remove_think_blocks(response)
        for pattern_str in self.pattern_strs:
            if pattern_str == "BOXED_PATTERN":
                extracted = self.get_boxed(response)
                if extracted:
                    return extracted
            else:
                matches = re.findall(pattern_str, response, re.DOTALL)
                if matches:
                    return matches[-1].strip()
        return ""


class AnswerMatcher(ABC):
    @abstractmethod
    def __call__(self, extracted_answer: str, annotation: dict[str, Any]) -> bool:
        pass


class IdentityAnswerMatcher(AnswerMatcher):
    def __call__(self, extracted_answer: str, annotation: dict[str, Any]) -> bool:
        return extracted_answer.strip().replace(" ", "") == str(annotation["answer"]).strip().replace(" ", "")


class MathExprAnswerMatcher(AnswerMatcher):
    def __call__(self, extracted_answer: str, annotation: dict[str, Any]) -> bool:
        return math_equal(extracted_answer, annotation["answer"])


Gsm8kAnswerMatcher = AimeAnswerMatcher = DapomathAnswerMatcher = IdentityAnswerMatcher
Math500AnswerMatcher = MathExprAnswerMatcher


class AnswerSimplifier(ABC):
    @abstractmethod
    def __call__(self, extracted: str) -> str:
        pass


class CountdownAnswerSimplifier(AnswerSimplifier):
    def __call__(self, extracted: str) -> str:
        answer = convert_countdown_answer(extracted)
        return str(eval_countdown_equation(answer))


class ExprAnswerSimplifier(AnswerSimplifier):
    def __call__(self, extracted: str) -> str:
        return simplify_math_expr(extracted)


Gsm8kAnswerSimplifier = AimeAnswerSimplifier = DapomathAnswerSimplifier = Math500AnswerSimplifier = (
    ExprAnswerSimplifier
)


class CountdownAnswerMatcher(AnswerMatcher):
    def __call__(self, extracted_answer: str, annotation: dict[str, Any]) -> bool:
        target = annotation["target"]
        numbers = annotation["numbers"]
        equation = convert_countdown_answer(extracted_answer)

        # Validate equation uses correct numbers
        if not validate_countdown_equation(equation, numbers):
            return False

        # Evaluate equation
        try:
            result = eval_countdown_equation(equation)
            if result is None or abs(result - target) > 1e-5:
                return False
            return True
        except Exception:  # noqa: E722
            return False
