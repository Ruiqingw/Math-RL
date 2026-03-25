import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
from transformers import AutoTokenizer
from reward_fn import PRMClassifier, score_steps, compute_reward
from step_splitter import split_into_steps
from verifier_prompt import format_verifier_prompt

MODEL_PATH = "/root/autodl-tmp/prm_grpo/verifier_cls/checkpoint-2000"
DEVICE = "cuda"

print("Loading model and tokenizer...")
model = PRMClassifier.from_pretrained(MODEL_PATH, device=DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Done.\n")

# Raw solutions stay paragraph-style; the runtime path normalizes them into
# [Previous steps] + [Current step] prompts before verifier scoring.
# Same problem, 3 variants: all-correct / wrong last step / wrong from step 2

PROBLEM = "What is the sum of all integer values of $n$ such that $\\frac{20}{2n - 1}$ is an integer?"

problems = [
    {
        "name": "① All steps correct  (gold_answer=2, answer=2)",
        "solution": (
            "We need $2n-1$ to be an odd divisor of 20.\n\n"
            "The odd divisors of 20 are $\\pm 1$ and $\\pm 5$.\n\n"
            "Setting $2n-1 = 1$ gives $n = 1$. Setting $2n-1 = -1$ gives $n = 0$. "
            "Setting $2n-1 = 5$ gives $n = 3$. Setting $2n-1 = -5$ gives $n = -2$.\n\n"
            "The sum is $1 + 0 + 3 + (-2) = \\boxed{2}$."
        ),
        "gold_answer": "2",
        # model's "answer" for r_final check — same as gold → correct
        "answer_for_r_final": "2",
    },
    {
        "name": "② Wrong last step only  (gold_answer=2, answer=1)",
        "solution": (
            "We need $2n-1$ to be an odd divisor of 20.\n\n"
            "The odd divisors of 20 are $\\pm 1$ and $\\pm 5$.\n\n"
            "Setting $2n-1 = 1$ gives $n = 1$. Setting $2n-1 = -1$ gives $n = 0$. "
            "Setting $2n-1 = 5$ gives $n = 3$. Setting $2n-1 = -5$ gives $n = -3$.\n\n"
            "The sum is $1 + 0 + 3 + (-3) = \\boxed{1}$."
        ),
        "gold_answer": "2",
        "answer_for_r_final": "1",
    },
    {
        "name": "③ Wrong from step 2  (gold_answer=2, answer=0)",
        "solution": (
            "We need $2n-1$ to be an odd divisor of 20.\n\n"
            "The odd divisors of 20 are $\\pm 2$ and $\\pm 10$.\n\n"
            "Setting $2n-1 = 2$ gives $n = 1.5$, not integer. "
            "Setting $2n-1 = -2$ gives $n = -0.5$, not integer. "
            "Setting $2n-1 = 10$ gives $n = 5.5$, not integer.\n\n"
            "There are no integer solutions, so the answer is $\\boxed{0}$."
        ),
        "gold_answer": "2",
        "answer_for_r_final": "0",
    },
]

for prob in problems:
    print(f"{'='*60}")
    print(f"{prob['name']}")
    steps = split_into_steps(prob['solution'])
    print(f"  Steps ({len(steps)}): {[s[:40]+'...' for s in steps]}")
    print("  Final-step verifier prompt:")
    print(format_verifier_prompt(PROBLEM, steps).replace("\n", "\n    "))

    step_scores = score_steps(
        problem=PROBLEM,
        steps=steps,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    print(f"  Step scores: {[f'{s:.3f}' for s in step_scores]}")

    # compute_reward uses gold_answer to check r_final internally
    # but it also extracts the model's answer from solution via \boxed{}
    r, info = compute_reward(
        problem=PROBLEM,
        solution=prob['solution'],
        gold_answer=prob['gold_answer'],
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
    )
    print(f"  r_final={info['r_final']:.1f}  r_avg_step={info['r_avg_step']:.3f}  r_first_error={info['r_first_error']:.3f}")
    print(f"  TOTAL REWARD: {r:.4f}")
    print()
