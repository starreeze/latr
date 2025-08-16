import re
from math import isclose

import regex
from func_timeout import func_timeout
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def convert_countdown_answer(answer: str) -> str:
    answer = re.sub(r" |\n", "", answer)
    answer = answer.split("=")[0]
    answer = answer.replace(r"\div", "/").replace(r"\times", "*")
    answer = answer.replace(r"\left(", "(").replace(r"\right)", ")")
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    return answer


def eval_countdown_equation(equation_str: str) -> float | None:
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def validate_countdown_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def _parse_latex(s):
    for f in [parse_latex, parse_expr]:
        try:
            return f(s.replace("\\\\", "\\"))
        except Exception:
            try:
                return f(s)
            except Exception:
                pass
    raise ValueError(f"Invalid latex: {s}")


def simplify_math_expr(expr: str) -> str:
    expr = _parse_latex(expr)
    return str(simplify(expr))


# the following code is adapted from https://github.com/GAIR-NLP/LIMO/blob/main/eval/utils/grader.py


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp[-1]
    else:
        pred = pred.strip().strip(".")
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    raise ValueError(f"Invalid number: {num}")


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


single_choice_patterns = [
    r"^\(A\)",
    r"^\(B\)",
    r"^\(C\)",
    r"^\(D\)",
    r"^\(E\)",  # (A) (B) (C) (D) (E)
    r"^A\.",
    r"^B\.",
    r"^C\.",
    r"^D\.",
    r"^E\.",  # A. B. C. D. E.
    r"^A\)",
    r"^B\)",
    r"^C\)",
    r"^D\)",
    r"^E\)",  # A) B) C) D) E)
    r"^\*\*A\*\*",
    r"^\*\*B\*\*",
    r"^\*\*C\*\*",
    r"^\*\*D\*\*",
    r"^\*\*E\*\*",  # **A** **B** **C** **D** **E**
    r"^A:",
    r"^B:",
    r"^C:",
    r"^D:",
    r"^E:",  # A: B: C: D: E:
]


def math_equal(
    prediction: str,
    reference: str,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: int = 0,
    depth: int = 0,
    max_depth: int = 5,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    if depth > max_depth:
        return False

    if prediction.strip().lower() == reference.strip().lower():
        return True
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    for pattern in single_choice_patterns:
        if regex.match(pattern, prediction):
            # Remove the pattern from the beginning of the prediction and strip the result
            prediction_cleaned = regex.sub(pattern, "", prediction, count=1).strip()
            # Recursively call math_equal to check if the cleaned prediction matches the reference
            if math_equal(
                prediction_cleaned,
                reference,
                include_percentage,
                is_close,
                timeout=timeout,
                depth=depth + 1,
                max_depth=max_depth,
            ):
                return True

    reference = re.sub(r"\\text\{([^}]*)\}", r"\1", reference)
    reference = re.sub(r"\\left([\{\[\(])", r"\1", reference)
    reference = re.sub(r"\\right([\}\]\)])", r"\1", reference)
    prediction = re.sub(r"\\text\{([^}]*)\}", r"\1", prediction)
    prediction = re.sub(r"\\left([\{\[\(])", r"\1", prediction)
    prediction = re.sub(r"\\right([\}\]\)])", r"\1", prediction)

    if "," in prediction and "," in reference:
        # 按逗号分割并去除空格
        pred_parts = [part.strip() for part in prediction.split(",")]
        ref_parts = [part.strip() for part in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            # 对两个列表排序后逐个比较，使用 math_equal 递归判断是否相等
            pred_parts_sorted = sorted(pred_parts)
            ref_parts_sorted = sorted(ref_parts)

            if all(
                math_equal(
                    pred_parts_sorted[i],
                    ref_parts_sorted[i],
                    include_percentage,
                    is_close,
                    timeout=timeout,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                for i in range(len(pred_parts_sorted))
            ):
                return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_result = [ref_num]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(pred_num, item):
                            return True
                    else:
                        if item == pred_num:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal

    ## pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i],
                        ref_parts[i],
                        include_percentage,
                        is_close,
                        timeout=timeout,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                                timeout=timeout,
                                depth=depth + 1,
                                max_depth=max_depth,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(
            prediction.split("=")[1],
            reference,
            include_percentage,
            is_close,
            timeout=timeout,
            depth=depth + 1,
            max_depth=max_depth,
        ):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(
            prediction,
            reference.split("=")[1],
            include_percentage,
            is_close,
            timeout=timeout,
            depth=depth + 1,
            max_depth=max_depth,
        ):
            return True

    if timeout > 0:
        if func_timeout(timeout, symbolic_equal, (prediction, reference)):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))

    # return isclose(reference, prediction, rel_tol=1e-4)
    return isclose(reference, prediction, abs_tol=1e-4)


def symbolic_equal(a, b):
    a = _parse_latex(a)
    b = _parse_latex(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def equal_level(a: str, b: str) -> int:
    "2: completely equal, 0: completely different, 1: not completely equal but eval to the same"
    if a == b:
        return 2
    if not a or not b:
        return 0
    try:
        sa = _parse_latex(a)
        sb = _parse_latex(b)
    except Exception:
        return 0
    if sa == sb:
        return 2
    try:
        if sa.equals(sb) or simplify(sa - sb) == 0:
            return 1
    except Exception:
        pass
    return 0
