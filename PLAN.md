# Implementation Plan — Gemma 4 12B on AWS g6.xlarge via vLLM

## Goal

Deploy `google/gemma-4-12B-it-qat-w4a16-ct` on single g6.xlarge (NVIDIA L4, 24GB VRAM) using vLLM as OpenAI-compatible inference server. Ship as GitHub-ready community repo with quickstart, benchmark results, cost table.

## Requirements

- Model fits in 24GB VRAM with headroom for concurrent requests
- Server survives ≥4 concurrent users at 32K context
- Python client (`main.py`) hits `localhost:8000` via OpenAI SDK
- Benchmark captures tok/s, TTFT, max-context-before-OOM
- Repo self-contained: clone → README → running server, no tribal knowledge

## Constraints

- W4A16 weights **AND** FP8 KV cache **AND** `--max-model-len 32768` — drop any one, OOM
- L4 sm_89 needs CUDA ≥12.1, driver ≥525
- vLLM ≥0.8.5 required for `--reasoning-parser gemma4` + `--tool-call-parser gemma4`
- HF model gated — license accept + token required before download
- FP8 KV cache can silently fall back to FP16 on wrong CUDA build — must verify in logs

## File Map

```
gemma_4_12b_quantization/
├── ANALYSIS.md             (existing — deployment rationale)
├── ARCHITECTURE.md         (new — system architecture)
├── PLAN.md                 (this file)
├── README.md               (new — quickstart)
├── main.py                 (replace boilerplate — OpenAI client demo)
├── benchmark.py            (new — tok/s, TTFT, OOM scan, concurrency)
├── serve.sh                (new — vllm launch wrapper)
├── setup.sh                (new — env bootstrap on EC2)
├── requirements.txt        (new — pinned deps)
├── pyproject.toml          (new — uv-friendly)
├── .env.example            (new — HF_TOKEN placeholder)
└── .gitignore              (new — exclude .env, .venv, hf_cache)
```

## Phases

### Phase 1 — EC2 provisioning (docs only)

- AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)` — ships CUDA 12.1 + driver 525+
- Instance: g6.xlarge, 80GB gp3 root (7GB weights + 10GB HF cache + 20GB vLLM/torch + 10GB OS + 15GB headroom)
- Security group: 22 SSH only. Keep 8000 localhost. Open 8000 only if exposing API externally
- Verify post-boot: `nvidia-smi` → driver ≥525, CUDA ≥12.1
- Document Spot ($0.27/hr) vs On-Demand ($0.80/hr) trade-off

### Phase 2 — Environment setup

- `setup.sh`: install `uv`, create `.venv` (Python 3.11), install vLLM CUDA 12.1 wheel, install `openai httpx rich python-dotenv huggingface_hub`
- `requirements.txt`: pin `vllm==0.8.5`, `openai>=1.30`, `httpx`, `rich`, `python-dotenv`, `huggingface_hub`
- `pyproject.toml`: PEP 517 metadata, mirrors requirements.txt for `uv sync`
- `.env.example`: `HF_TOKEN=hf_...` with comment on gated model access
- `.gitignore`: `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `hf_cache/`, `benchmark_results.csv` if user choice

### Phase 3 — HF auth + model download

- README Step 1: HF account → visit model page → accept Google license → generate token (read scope) → put in `.env`
- `setup.sh` sanity check: `huggingface-cli whoami` — fail fast if token missing
- Optional pre-download: `huggingface-cli download google/gemma-4-12B-it-qat-w4a16-ct` (~7GB, 5-10 min on EC2)

### Phase 4 — vLLM server launch

- `serve.sh`:
  ```bash
  source .venv/bin/activate
  source .env
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  vllm serve google/gemma-4-12B-it-qat-w4a16-ct \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --enable-auto-tool-choice \
    --mm-processor-kwargs '{"max_soft_tokens": 280}' \
    --limit-mm-per-prompt '{"image": 4, "audio": 1}'
  ```
- Launch via `tmux new-session -d -s vllm 'bash serve.sh 2>&1 | tee vllm.log'`
- Health check loop: `curl -s localhost:8000/health` retry until ready
- Validation: grep `vllm.log` for `"FP8 KV cache enabled"` — confirms no silent fallback

### Phase 5 — Python client

- `main.py`: 4 functions, sequential demo
  1. Basic text completion (server alive check)
  2. Tool call (exercises `--tool-call-parser gemma4`)
  3. Reasoning/thinking mode (exercises `--reasoning-parser gemma4`)
  4. Multimodal: text + image URL (exercises mm limits)
- `openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")`

### Phase 6 — Benchmark harness

- `benchmark.py` measures:
  - **Throughput**: single request, output tokens / elapsed
  - **TTFT**: streaming API, send → first chunk
  - **Max context**: binary search 4K→32K in 4K steps, stop at 500/OOM
  - **Concurrency**: 1/2/4/8 async requests via `httpx.AsyncClient`, aggregate tok/s + per-req TTFT
- Output: `rich` table + `benchmark_results.csv` (committed)
- All requests `timeout=60` — vLLM OOM can hang, not always 500

### Phase 7 — Cost calculation

- Formula: `$/1M tok = ($/hr) / (tok/s × 3600) × 1_000_000`
- Fill after benchmark. Compare vs GPT-4o-mini ($0.60/1M), Gemini 2.0 Flash ($0.40/1M)
- Both Spot + On-Demand rows

### Phase 8 — README

- Sections:
  1. What this is (2 sentences)
  2. Hardware requirements
  3. Prerequisites — **HF license accept FIRST**, before any code
  4. Quickstart: launch EC2 → SSH → clone → `setup.sh` → `.env` → `serve.sh` → `python main.py`
  5. vLLM flags explained (link ANALYSIS.md for deep dive)
  6. Benchmark results (filled from Phase 6)
  7. Cost table (filled from Phase 7)
  8. Known limits: 32K ctx cap, mm payload limits, no multi-GPU on g6.xlarge
  9. Troubleshooting: OOM startup → driver version; 403 download → license; tool calls broken → vLLM version

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| vLLM lacks gemma4 parsers | High | Pin `vllm==0.8.5` in requirements.txt; document min version |
| OOM on startup | High | `--gpu-memory-utilization 0.92` + FP8 KV; fallback `--max-model-len 16384` documented |
| OOM mid-request (concurrent + max ctx) | Medium | Benchmark reveals true ceiling; document |
| FP8 KV silent fallback to FP16 | High | Grep startup log for confirmation line; document check |
| HF 403 blocks first-time user | High | README Step 1 = license accept; `huggingface-cli whoami` in setup.sh |
| DL AMI ID rotates | Medium | README uses search string, not ID |
| Multimodal OOM | Medium | Both mm flags from day one |
| HF_TOKEN committed | High | `.gitignore` + `.env.example` pattern |
| Benchmark hangs on OOM | Medium | `timeout=60` on all requests |
| Spot interruption mid-benchmark | Low | Recommend On-Demand for benchmarks |

## Success Criteria

- [ ] `nvidia-smi` shows L4, driver ≥525, CUDA ≥12.1
- [ ] `vllm serve` starts no OOM, log confirms FP8 KV active
- [ ] `curl localhost:8000/health` → `{"status":"ok"}`
- [ ] `python main.py` completes all 4 examples
- [ ] `python benchmark.py` produces CSV
- [ ] README quickstart works for zero-context user
- [ ] No secrets in git
- [ ] Cost table has real numbers, no placeholders

## Implementation Order

```
Phase 1 (docs)       → README sections drafted
Phase 2 (env)        → setup.sh, requirements.txt, pyproject.toml, .env.example, .gitignore
Phase 3 (HF auth)    → README + setup.sh additions
Phase 4 (serve)      → serve.sh + README validation
Phase 5 (client)     → main.py replaces boilerplate
Phase 6 (benchmark)  → benchmark.py
Phase 7 (cost)       → README section (after Phase 6)
Phase 8 (README)     → assemble final
```

Phases 1-3 writable locally. Phases 4-7 need live EC2. Phase 8 needs Phase 6 results.

## Complexity Estimate

Medium. ~6-10 hr engineering + ~2 hr EC2 time.