"""
step_splitter.py — Split a math solution string into reasoning steps.

Strategy:
  1. Split on double newlines (paragraph breaks) first.
  2. Within a long paragraph (>300 chars), try to further split on
     sentence-ending punctuation followed by a newline or a capital letter.
  3. Filter out chunks that are too short to be meaningful.
"""

import re
from typing import List


def split_into_steps(text: str, min_chars: int = 20, max_chars_per_step: int = 300) -> List[str]:
    """
    Split a math solution into a list of step strings.

    Args:
        text: The full solution text.
        min_chars: Discard any chunk shorter than this.
        max_chars_per_step: Paragraphs longer than this get further split.

    Returns:
        List of step strings, at least one element.
    """
    if not text or not text.strip():
        return [""]

    # Step 1: split on double (or more) newlines
    paragraphs = re.split(r'\n{2,}', text.strip())

    steps: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars_per_step:
            if len(para) >= min_chars:
                steps.append(para)
        else:
            # Step 2: split long paragraphs on sentence boundaries
            # Matches: period/!/? followed by newline OR followed by whitespace+Capital/$+\
            sub = re.split(r'(?<=[.!?])\n|(?<=[.!?])\s+(?=[A-Z\$\\])', para)
            for chunk in sub:
                chunk = chunk.strip()
                if len(chunk) >= min_chars:
                    steps.append(chunk)

    # Fallback: never return empty list
    if not steps:
        stripped = text.strip()
        steps = [stripped] if stripped else [""]

    return steps


def _is_escaped(text: str, idx: int) -> bool:
    """Return True if text[idx] is preceded by an odd number of backslashes."""
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return (backslashes % 2) == 1


def _extract_braced_content(text: str, open_brace_idx: int) -> str:
    """
    Extract balanced {...} content starting at text[open_brace_idx] == '{'.
    Returns empty string if the brace sequence is incomplete.
    """
    depth = 0
    content_chars = []

    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]

        if ch == "{" and not _is_escaped(text, idx):
            depth += 1
            if depth > 1:
                content_chars.append(ch)
            continue

        if ch == "}" and not _is_escaped(text, idx):
            if depth == 0:
                return ""
            depth -= 1
            if depth == 0:
                return "".join(content_chars)
            content_chars.append(ch)
            continue

        if depth >= 1:
            content_chars.append(ch)

    return ""


def extract_boxed_answer(text: str) -> str:
    """
    Extract the last \\boxed{...} answer from a solution string.
    Uses balanced-brace scanning so nested LaTeX like \\boxed{\\frac{1}{2}}
    is handled correctly. Returns empty string if not found.
    """
    matches = []
    for boxed_match in re.finditer(r'\\boxed\s*\{', text):
        open_brace_idx = boxed_match.end() - 1
        content = _extract_braced_content(text, open_brace_idx)
        if content:
            matches.append(content.strip())
    return matches[-1] if matches else ""


if __name__ == "__main__":
    # Quick smoke test
    sample = """Let $x$ be the number of apples.  We know $x > 0$.

Since each basket holds 5 apples, we have $x = 5k$ for some integer $k$.

Substituting back: $5k = 20$, so $k = 4$.

Therefore $x = \\boxed{20}$."""

    steps = split_into_steps(sample)
    print(f"Found {len(steps)} steps:")
    for i, s in enumerate(steps):
        print(f"  [{i+1}] {s[:80]}...")
    print("\nExtracted answer:", extract_boxed_answer(sample))
    print("Nested boxed answer:", extract_boxed_answer(r"Final: \boxed{\frac{1}{2}}"))
