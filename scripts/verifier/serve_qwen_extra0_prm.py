#!/usr/bin/env python3
"""Serve a Qwen-style extra0 token-classification PRM over a local HTTP API."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Any, List, Sequence

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from qwen_extra0_prm import load_extra0_prm, score_steps  # noqa: E402


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "token_prm_runs/extra0-prm/final"


class ScoreItem(BaseModel):
    problem: str
    steps: List[str] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    items: List[ScoreItem]
    max_length: int | None = None


class ScoreResult(BaseModel):
    score: float
    step_scores: List[float]
    n_steps: int


class ScoreResponse(BaseModel):
    scores: List[ScoreResult]
    request_count: int
    batch_size: int
    latency_ms: float
    error_count: int


class HealthResponse(BaseModel):
    ok: bool
    model_path: str
    request_count: int
    error_count: int


def aggregate_step_scores(step_scores: Sequence[float]) -> float:
    if not step_scores:
        return 0.0
    return float(min(step_scores))


class PRMServerState:
    def __init__(self) -> None:
        self.model_path = ""
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        self.max_length = 1536
        self.lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0


STATE = PRMServerState()
app = FastAPI(title="Math-RL extra0 PRM reward server")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=STATE.model is not None and STATE.tokenizer is not None,
        model_path=STATE.model_path,
        request_count=STATE.request_count,
        error_count=STATE.error_count,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if STATE.model is None or STATE.tokenizer is None:
        raise HTTPException(status_code=503, detail="PRM model is not loaded")

    start_time = time.perf_counter()
    max_length = int(request.max_length or STATE.max_length)
    results: list[ScoreResult] = []

    try:
        with STATE.lock:
            for item in request.items:
                step_scores = score_steps(
                    item.problem,
                    item.steps,
                    STATE.model,
                    STATE.tokenizer,
                    device=STATE.device,
                    max_length=max_length,
                    require_all_steps=False,
                )
                results.append(
                    ScoreResult(
                        score=aggregate_step_scores(step_scores),
                        step_scores=[float(score) for score in step_scores],
                        n_steps=len(item.steps),
                    )
                )
            STATE.request_count += 1
    except Exception as exc:  # pragma: no cover - exercised during server runtime
        STATE.error_count += 1
        logger.exception("PRM scoring request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return ScoreResponse(
        scores=results,
        request_count=STATE.request_count,
        batch_size=len(request.items),
        latency_ms=latency_ms,
        error_count=STATE.error_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    device_map = None if args.device == "cpu" else "auto"
    logger.info("Loading extra0 PRM server model: path=%s device=%s", args.model_path, args.device)
    model, tokenizer, _ = load_extra0_prm(
        args.model_path,
        device_map=device_map,
        dtype=dtype_from_arg(args.dtype),
    )
    STATE.model_path = args.model_path
    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.device = args.device
    STATE.max_length = args.max_length
    logger.info("Starting PRM reward server on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
