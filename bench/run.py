"""Concurrency benchmark for llama.cpp / vLLM OpenAI-compatible endpoint.

Hits /v1/chat/completions at each concurrency level, reports per-user and
aggregate throughput.

Env vars:
  LLAMA_URL          base URL (default http://llama:8000)
  VLLM_API_KEY       bearer token (required)
  CONCURRENCY_LEVELS comma list, default "1,4,8,16,32"
  MAX_TOKENS         per request, default 300
  MODEL              model name, default "gemma-4-12b"
  PROMPT             user prompt, default "Count from 1 to 100, one per line."
  WARMUP             1 to send single warmup request first (default 1)
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("missing httpx: pip install httpx")


URL = os.environ.get("LLAMA_URL", "http://llama:8000").rstrip("/") + "/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY")
LEVELS = [int(x) for x in os.environ.get("CONCURRENCY_LEVELS", "1,4,8,16,32").split(",")]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))
MODEL = os.environ.get("MODEL", "gemma-4-12b")
PROMPT = os.environ.get("PROMPT", "Count from 1 to 100, one number per line.")
WARMUP = os.environ.get("WARMUP", "1") == "1"

if not KEY:
    sys.exit("VLLM_API_KEY not set")

PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOKENS,
    "chat_template_kwargs": {"thinking": False},
}
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


async def one(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, headers=HEADERS, json=PAYLOAD, timeout=600)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return {"status": r.status_code, "elapsed_s": elapsed, "error": r.text[:200]}
        j = r.json()
        t = j.get("timings", {})
        return {
            "status": 200,
            "elapsed_s": elapsed,
            "tok_per_s": float(t.get("predicted_per_second", 0)),
            "tokens": int(t.get("predicted_n", 0)),
        }
    except Exception as e:
        return {"status": -1, "elapsed_s": time.perf_counter() - t0, "error": repr(e)}


async def burst(n: int) -> dict:
    limits = httpx.Limits(max_connections=n + 4, max_keepalive_connections=n + 4)
    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(client) for _ in range(n)])
        wall = time.perf_counter() - t0

    ok = [r for r in results if r.get("status") == 200]
    fail = [r for r in results if r.get("status") != 200]
    tps = [r["tok_per_s"] for r in ok]
    elapsed = [r["elapsed_s"] for r in ok]
    toks = sum(r["tokens"] for r in ok)

    return {
        "concurrency": n,
        "ok": len(ok),
        "fail": len(fail),
        "wall_s": wall,
        "per_user_tps_avg": statistics.mean(tps) if tps else 0,
        "per_user_tps_min": min(tps) if tps else 0,
        "per_user_tps_max": max(tps) if tps else 0,
        "p50_latency_s": statistics.median(elapsed) if elapsed else 0,
        "aggregate_tps": toks / wall if wall > 0 else 0,
        "total_tokens": toks,
        "first_error": fail[0].get("error", "") if fail else "",
    }


async def main() -> None:
    print(f"target: {URL}")
    print(f"model:  {MODEL}")
    print(f"prompt: {PROMPT!r}")
    print(f"max_tokens: {MAX_TOKENS}")
    print(f"levels: {LEVELS}")
    print()

    if WARMUP:
        print("warmup ...")
        async with httpx.AsyncClient() as c:
            r = await one(c)
            if r["status"] != 200:
                sys.exit(f"warmup failed: {r}")
        print(f"warmup ok: {r['tok_per_s']:.1f} tok/s\n")

    header = f"{'conc':>5} {'ok':>3} {'fail':>4} {'wall_s':>8} {'p50_lat':>9} {'tps/user_avg':>13} {'tps/user_min':>13} {'tps/user_max':>13} {'agg_tps':>9}"
    print(header)
    print("-" * len(header))
    for n in LEVELS:
        r = await burst(n)
        print(
            f"{r['concurrency']:>5} {r['ok']:>3} {r['fail']:>4} "
            f"{r['wall_s']:>8.2f} {r['p50_latency_s']:>9.2f} "
            f"{r['per_user_tps_avg']:>13.2f} {r['per_user_tps_min']:>13.2f} {r['per_user_tps_max']:>13.2f} "
            f"{r['aggregate_tps']:>9.2f}"
        )
        if r["fail"]:
            print(f"      first error: {r['first_error']}")


if __name__ == "__main__":
    asyncio.run(main())
