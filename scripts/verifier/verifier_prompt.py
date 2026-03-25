"""
Shared verifier prompt formatting.

The verifier judges whether the current step is mathematically correct
given the problem and any previous steps.
"""

from typing import List


def format_verifier_prompt(problem: str, steps: List[str]) -> str:
    """
    Build the prompt fed to the verifier for step k (the last element of steps).

    The current step is separated from previous context so the classifier can
    focus on the step being judged instead of treating the whole solution as
    one undifferentiated block.
    """
    if not steps:
        raise ValueError("steps must contain at least one step")

    previous_steps = steps[:-1]
    current_step = steps[-1]

    if previous_steps:
        previous_text = "\n".join(
            f"Step {i+1}: {step}" for i, step in enumerate(previous_steps)
        )
    else:
        previous_text = "(none)"

    current_text = f"Step {len(steps)}: {current_step}"

    return (
        f"[Problem]\n{problem}\n\n"
        f"[Previous steps]\n{previous_text}\n\n"
        f"[Current step]\n{current_text}\n\n"
        f"Is the current step mathematically correct given the previous steps?\n"
        f"Answer:"
    )
