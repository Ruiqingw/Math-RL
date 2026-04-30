"""
step_splitter.py — Split a math solution string into reasoning steps.

Strategy:
  1. Prefer explicit structure: blank lines and numbered step lines.
  2. Keep sentence splitting as a fallback only when no structure is present.
  3. Merge very short structured chunks instead of dropping them.
"""

import re
from typing import List


_NUMBERED_STEP_RE = re.compile(
    r"^\s*(?:\(?\d+[\).:]|step\s+\d+[\).:-])\s+",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\\])|(?<=[.!?])\n+")


def _nonempty_chunks(chunks: List[str]) -> List[str]:
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _split_numbered_lines(block: str) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _NUMBERED_STEP_RE.match(line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())
    return _nonempty_chunks(chunks)


def _structured_split(text: str) -> List[str]:
    paragraphs = _nonempty_chunks(re.split(r"(?:\r?\n\s*){2,}", text.strip()))
    chunks: List[str] = []
    for paragraph in paragraphs:
        chunks.extend(_split_numbered_lines(paragraph))
    return _nonempty_chunks(chunks)


def _merge_short_steps(steps: List[str], min_chars: int) -> List[str]:
    if min_chars <= 0 or len(steps) <= 1:
        return _nonempty_chunks(steps)

    merged: List[str] = []
    pending = ""
    for step in steps:
        if len(step) >= min_chars:
            if pending:
                step = f"{pending}\n{step}"
                pending = ""
            merged.append(step)
        elif merged:
            merged[-1] = f"{merged[-1]}\n{step}".strip()
        else:
            pending = f"{pending}\n{step}".strip() if pending else step

    if pending:
        merged.append(pending)
    return _nonempty_chunks(merged)


def _sentence_fallback(text: str, min_chars: int) -> List[str]:
    chunks = _nonempty_chunks(_SENTENCE_SPLIT_RE.split(text.strip()))
    return _merge_short_steps(chunks, min_chars)


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

    stripped = text.strip()
    structured_raw_steps = _structured_split(stripped)
    if len(structured_raw_steps) > 1 and any(_NUMBERED_STEP_RE.match(step) for step in structured_raw_steps):
        return structured_raw_steps

    structured_steps = _merge_short_steps(structured_raw_steps, min_chars)
    if len(structured_steps) > 1:
        return structured_steps
    if structured_steps and len(structured_steps[0]) <= max_chars_per_step:
        return structured_steps

    steps = _sentence_fallback(stripped, min_chars)
    if not steps:
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
    numbered = "1. Let x be the value.\n2. Then x + 1 = 3.\n3. Therefore x = \\boxed{2}."
    print("\nNumbered steps:", split_into_steps(numbered))
    print("\nExtracted answer:", extract_boxed_answer(sample))
    print("Nested boxed answer:", extract_boxed_answer(r"Final: \boxed{\frac{1}{2}}"))
