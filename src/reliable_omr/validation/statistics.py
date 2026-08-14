"""Exact binomial confidence bounds for conservative release decisions."""

import math
from typing import Any, Dict


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""

    max_iterations = 300
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError(
        "Incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    term = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - term * _beta_continued_fraction(b, a, 1.0 - x) / b


def clopper_pearson_lower_bound(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> float:
    """Return the exact one-sided Clopper-Pearson lower confidence bound."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if successes == 0 or total == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    if successes == total:
        return math.exp(math.log(alpha) / total)

    low = 0.0
    high = successes / float(total)
    beta_b = total - successes + 1
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _regularized_beta(midpoint, successes, beta_b) < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def zero_error_sample_requirement(
    target_accuracy: float, confidence_level: float = 0.95
) -> int:
    """Minimum sample count whose zero-error lower bound reaches the target."""

    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must be in (0, 1)")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = 1.0 - confidence_level
    required = int(math.ceil(math.log(alpha) / math.log(target_accuracy)))
    while math.exp(math.log(alpha) / required) < target_accuracy:
        required += 1
    return required


def accuracy_gate(
    correct: int,
    total: int,
    target_accuracy: float = 0.999,
    confidence_level: float = 0.95,
    minimum_samples: int = 1,
) -> Dict[str, Any]:
    """Evaluate a correctness target without passing on a point estimate."""

    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if correct < 0 or total < 0 or correct > total:
        raise ValueError("correct must be between zero and total")
    required_zero_error = zero_error_sample_requirement(
        target_accuracy, confidence_level
    )
    point_accuracy = correct / float(total) if total else None
    lower_bound = clopper_pearson_lower_bound(
        correct, total, confidence_level
    )

    if total == 0:
        status = "insufficient_data"
        reason = "No auto-accepted answers have human ground truth."
    elif point_accuracy < target_accuracy:
        status = "fail"
        reason = "Observed auto-accepted accuracy is below the target."
    elif total < minimum_samples:
        status = "insufficient_data"
        reason = "Accepted sample count is below the configured minimum."
    elif total < required_zero_error:
        status = "insufficient_data"
        reason = (
            "Even zero observed errors cannot establish the target at this "
            "confidence level with the available sample count."
        )
    elif lower_bound >= target_accuracy:
        status = "pass"
        reason = "The one-sided exact lower confidence bound meets the target."
    else:
        status = "fail"
        reason = (
            "The one-sided exact lower confidence bound is below the target."
        )

    return {
        "status": status,
        "correct": correct,
        "errors": total - correct,
        "accepted_samples": total,
        "point_accuracy": point_accuracy,
        "target_accuracy": target_accuracy,
        "confidence_level": confidence_level,
        "one_sided_lower_bound": lower_bound,
        "method": "one-sided exact Clopper-Pearson binomial bound",
        "configured_minimum_samples": minimum_samples,
        "zero_error_samples_required": required_zero_error,
        "reason": reason,
    }
