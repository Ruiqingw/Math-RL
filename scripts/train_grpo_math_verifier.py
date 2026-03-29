#!/usr/bin/env python3
import os
import runpy
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.trl.train_grpo_math_verifier import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("scripts.trl.train_grpo_math_verifier", run_name="__main__")
