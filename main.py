"""Async client for local Gemma 4 12B server (llama.cpp / vLLM).

Usage:
  python main.py                            # 1 request, default prompt
  python main.py "Explain quantization."    # 1 request, custom prompt
  python main.py "Task X" --concurrency 16  # 16 parallel reqs, same prompt
  python main.py --concurrency 8 --max-tokens 500 "..."

Env:
  LLAMA_API_KEY    bearer token (required)
  VLLM_URL        base URL, default http://localhost:8000/v1
  MODEL_NAME      model alias, default gemma-4-12b
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from openai import AsyncOpenAI

MODEL = os.getenv("MODEL_NAME", "gemma-4-12b")
BASE_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")


def build_client() -> AsyncOpenAI:
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        sys.exit("LLAMA_API_KEY not set. export it or source .env first.")
    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key)


async def one(client: AsyncOpenAI, idx: int, prompt: str, max_tokens: int, temperature: float, thinking: bool) -> dict:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"thinking": thinking}},
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    return {
        "idx": idx,
        "elapsed_s": elapsed,
        "content": msg.content or "",
        "reasoning": getattr(msg, "reasoning_content", "") or "",
        "tokens": resp.usage.completion_tokens if resp.usage else 0,
    }


async def run(prompt: str, concurrency: int, max_tokens: int, temperature: float, thinking: bool) -> None:
    client = build_client()
    print(f"target: {BASE_URL}  model: {MODEL}  concurrency: {concurrency}  max_tokens: {max_tokens}  thinking: {thinking}")
    print(f"prompt: {prompt!r}\n")

    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        one(client, i, prompt, max_tokens, temperature, thinking)
        for i in range(concurrency)
    ])
    wall = time.perf_counter() - t0

    for r in results:
        tps = r["tokens"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
        print(f"[{r['idx']:2d}] {r['elapsed_s']:.2f}s · {r['tokens']:4d} tok · {tps:.1f} tok/s")
        if concurrency == 1:
            if r["reasoning"]:
                print(f"\n--- reasoning ---\n{r['reasoning']}")
            print(f"\n--- answer ---\n{r['content']}\n")

    if concurrency > 1:
        total_tokens = sum(r["tokens"] for r in results)
        agg = total_tokens / wall if wall > 0 else 0
        avg_per_user = sum(r["tokens"] / r["elapsed_s"] for r in results if r["elapsed_s"] > 0) / len(results)
        print(f"\nwall: {wall:.2f}s · total tokens: {total_tokens} · agg: {agg:.1f} tok/s · per-user avg: {avg_per_user:.1f} tok/s")

    await client.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Async Gemma 4 12B client")
    p.add_argument("prompt", nargs="?", default="Explain INT4 quantization in two sentences.")
    p.add_argument("-c", "--concurrency", type=int, default=1, help="parallel requests (default 1)")
    p.add_argument("-n", "--max-tokens", type=int, default=512)
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.add_argument("--thinking", action="store_true", help="enable Gemma 4 thinking mode (slower, includes reasoning trace)")
    args = p.parse_args()

    if args.concurrency < 1:
        sys.exit("--concurrency must be >= 1")

    asyncio.run(run(args.prompt, args.concurrency, args.max_tokens, args.temperature, args.thinking))


if __name__ == "__main__":
    main()
