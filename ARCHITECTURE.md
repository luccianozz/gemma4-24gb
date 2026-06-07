# Architecture — Gemma 4 12B on L4

Default backend: **llama.cpp** (CUDA build). Alt backend: **vLLM** (blocked by upstream Gemma 4 bug — see [vLLM addendum](#vllm-alt-backend-blocked)).

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Client (host or remote)                                         │
│   └── OpenAI SDK → POST /v1/chat/completions  + Bearer token     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  AWS EC2 g6.xlarge  (Amazon Linux / Ubuntu, DL AMI)              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Docker container: gemma4-12b                            │     │
│  │   image: ghcr.io/ggml-org/llama.cpp:server-cuda         │     │
│  │   port  8000 (OpenAI-compatible REST + /metrics)        │     │
│  │   auth  --api-key=$VLLM_API_KEY (Bearer)                │     │
│  │                                                         │     │
│  │  ┌─────────────────────────────────────────────────┐    │     │
│  │  │ llama.cpp server                                │    │     │
│  │  │   continuous batching                           │    │     │
│  │  │   16 parallel slots × ~8K avg ctx               │    │     │
│  │  │   (--ctx-size 131072 total budget)              │    │     │
│  │  └─────────────────────────────────────────────────┘    │     │
│  └────────────────────────────┬────────────────────────────┘     │
│                               │                                  │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ NVIDIA L4 — 24GB GDDR6 (Ada Lovelace sm_89)             │     │
│  │   Weights GGUF Q4_K_M             ~7.5 GB               │     │
│  │   KV cache (Q8 quantized)         ~10-12 GB             │     │
│  │   Activations + scratch           ~2 GB                 │     │
│  │   Driver + CUDA runtime           ~2 GB                 │     │
│  │                                                         │     │
│  │ Kernels (Ada sm_89):                                    │     │
│  │   ├─ ggml CUDA Q4_K matmul (dequant + FP16 GEMM)        │     │
│  │   ├─ FlashAttention-2 (BF16/FP16 compute)               │     │
│  │   └─ Q8/Q4 KV cache (in-kernel dequant)                 │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Disk: 80GB gp3                                                  │
│   └─ llama_cache volume (~7.5GB GGUF + metadata)                 │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### Model layer

| Item | Value |
|------|-------|
| Checkpoint | `bartowski/gemma-4-12B-it-GGUF:Q4_K_M` (community Q4_K_M) |
| Alt checkpoint | `google/gemma-4-12B-it-qat-q4_0-gguf` (official QAT Q4_0) |
| Format | GGUF |
| Quantization | Q4_K_M weights (mixed 4/5/6-bit, ~4.85 bpw average) |
| KV cache | Q8_0 (both K and V) |
| Max context | 131072 tokens (configurable via `--ctx-size`) |
| Modalities | Text (image support via separate mmproj — out of current scope) |

### Runtime layer

| Item | Value |
|------|-------|
| Engine | llama.cpp server (CUDA build) |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| Matmul kernel | ggml Q4_K CUDA (per-block dequant → FP16 GEMM) |
| Attention | FlashAttention-2 (BF16 compute on Ada) |
| KV cache | Q8_0 in-place — halves KV memory vs FP16, negligible quality loss |
| Scheduler | Continuous batching, `--parallel 16` slots |
| API | OpenAI-compatible REST + `/metrics` Prometheus |
| Auth | `--api-key` → clients send `Authorization: Bearer <key>` |
| Process mgr | Docker `restart: unless-stopped` |

### Infra layer

| Item | Value |
|------|-------|
| Instance | AWS g6.xlarge |
| GPU | NVIDIA L4, 24GB, sm_89 |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x |
| Driver | ≥525 |
| CUDA | ≥12.1 |
| Container runtime | Docker + nvidia-container-toolkit |
| Disk | 80GB gp3 |
| Network | Default VPC, SG: 22 inbound only (port 8000 localhost) |

## Request Flow

```
1. Client → POST /v1/chat/completions + Authorization: Bearer <api_key>
2. llama.cpp server:
   ├─ verify bearer
   ├─ tokenize prompt (SentencePiece for Gemma)
   ├─ assign to a free slot OR queue if all 16 busy
   └─ continuous batching: this slot joins next forward pass batch
3. Forward pass (per generated token, batched across active slots):
   ├─ Q4_K matmul (dequant blocks → FP16 GEMM)
   ├─ Attention reads Q8 KV pages (dequant in kernel)
   └─ Sample next token per slot
4. Stream chunk → client (SSE)
5. Repeat until EOS / max_tokens / stop sequence
6. Free slot, return final usage stats
```

## VRAM Budget — 16 slots × 8K avg context

L4 hardware total = 24GB. Driver + CUDA runtime ~2GB outside llama.cpp.

| Component | VRAM | Notes |
|-----------|------|-------|
| NVIDIA driver + CUDA runtime | ~2 GB | Outside container |
| **llama.cpp budget** | **~22 GB** | |
| └─ Model weights (Q4_K_M) | ~7.5 GB | Static, loaded once |
| └─ Activations + scratch | ~2 GB | Per-step transient |
| └─ ggml workspace | ~0.5 GB | Buffers, KV indexing |
| └─ KV cache (Q8) available | **~12 GB** | 131K tok budget shared across 16 slots |
| **Hardware total** | **24 GB** | |

KV @ Q8 ≈ **~40 KB/token** for Gemma 12B. 12 GB budget ÷ 40 KB ≈ **~300K tokens total**, comfortable for `--ctx-size 131072` (131K).

### Measured throughput on g6.xlarge (1× L4)

Config: Gemma 4 12B Q4_K_M, Q8 KV, `--parallel 16 --ctx-size 131072 --flash-attn on`.

**Run A — short prompt (~20 tok), decode 300:**

| Conc | tps/user | agg | p50 lat |
|------|----------|-----|---------|
| 1 | 27.7 | 27 | 11.0s |
| 4 | 20.7 | 81 | 15.0s |
| 8 | 12.6 | 98 | 24.4s |
| **16** | **16.6** | **252** | **19.0s** ★ |
| 32 | 16.5 | 252 | 28.6s (queue) |

**Run B — medium prompt (~30 tok), decode 500:**

| Conc | tps/user | agg | p50 lat |
|------|----------|-----|---------|
| 1 | 27.6 | 27 | 18.3s |
| 4 | 19.6 | 77 | 25.8s |
| 8 | 12.1 | 95 | 41.9s |
| **16** | **16.1** | **248** | **32.3s** ★ |
| 24 | 14.6 | 158 | 32.6s (bad batching) |
| 32 | 16.1 | 245 | 48.9s |

**Run C — queue knee (200 decode):**

| Conc | tps/user | agg | p50 lat |
|------|----------|-----|---------|
| 16 | 17.0 | 244 | 13.1s |
| 20 | 17.4 | 172 | 12.6s (uneven) |
| 24 | 15.4 | 165 | 12.6s (uneven) |
| 32 | 16.7 | 251 | 19.1s |
| 48 | 16.8 | 251 | 25.5s |
| 64 | 16.8 | 250 | 32.0s |

**Run D — long prefill (2000 tok prompt), decode 100:**

| Conc | tps/user | agg | wall |
|------|----------|-----|------|
| 1 | 26.5 | 26 | 3.8s |
| 4 | 14.6 | 38 | 10.5s |
| 8 | 8.1 | 49 | 16.2s |
| 16 | 6.9 | **74** | 21.7s |

### Locked-in conclusions

| Finding | Number |
|---------|--------|
| **Aggregate ceiling** (short prompts) | **~250-257 tok/s** |
| **Sweet spot concurrency** | **16** (matches `--parallel 16`) |
| **Per-user tok/s @ sweet spot** | **~16-17** |
| Conc=8 droop (real, reproducible) | ~12 tps/user — avoid this batch size |
| Conc=20-24 worst-case (uneven slots) | ~165 agg — avoid |
| Past 16 (queue regime) | per-user stable ~16.7, wall scales linearly |
| Long prefill kills aggregate | 2000-tok prompt → 74 tps/user agg @ 16 conc |
| Queue depth formula | latency ≈ 13s × ceil(N/16) for 200-tok decode |

### Recommendations

- **Short Q&A workload (≤500 tok prompt + ≤500 tok decode)**: `--parallel 16`, default. Real capacity ~16 simultaneous at ~16 tps/user
- **Long-prompt workload (>1K tok prompt)**: drop to `--parallel 8`, enable prefix caching for repeated system prompts
- **Avoid** running close to 20-24 concurrent — uneven slot allocation degrades agg. Either stay ≤16 or accept queue at ≥32
- **300+ concurrent**: not viable on single L4. 19 batches deep → ~4 min wait for last user. Need 2× A100 80GB or 1× H100.

## Failure Modes

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| OOM at startup | Container crash before `/health` | Too-large `--ctx-size` × `--parallel` | Lower ctx-size or parallel; verify Q8 KV active |
| Slow per-user under load | Aggregate fine, individual <5 tok/s | Too many parallel slots | Reduce `--parallel` |
| 401 on requests | API rejects bearer | Wrong `VLLM_API_KEY` in client | Confirm `.env` matches server-side flag |
| 404 model name | `model="..."` mismatch | Client passing wrong name | Use `gemma-4-12b` (= `--alias` value) |
| Cold start ~5-10 min | First `up` slow | HF download (~7.5GB) | Persisted in `llama_cache` volume; warm restart fast |
| Container restart loop | Repeated crash | Bad flag or missing GPU | `docker compose logs llama` — check first error |

## Scaling Path (out of scope, documented)

| Need | Path |
|------|------|
| 30-50 concurrent at higher tok/s | g6.12xlarge (4× L4) + multi-process LB or vLLM TP |
| 100-300 concurrent at production tok/s | 2× A100 80GB or 1× H100, vLLM/TGI |
| Higher quality at same speed | Q5_K_M (~8.5GB) — costs ~1GB KV budget |
| Longer per-slot context | Drop slots: 4×32K or 2×64K |
| Image input | mmproj sidecar (Gemma 4 vision adapter) |
| Tool calling | llama.cpp grammars or post-process model output |

## Security Notes

- `HF_TOKEN` + `VLLM_API_KEY` only in `.env` (gitignored). Never in compose file or code
- `--api-key` enforces bearer on **all** `/v1/*` endpoints (no anon)
- `/health` + `/metrics` typically unauthenticated — keep port 8000 localhost-only or front with reverse proxy + IP allowlist for `/metrics`
- llama.cpp downloads GGUF from HF on first boot — verify checksum against `bartowski/gemma-4-12B-it-GGUF` HF page if integrity matters
- Output not sanitized — treat model text as untrusted before passing to downstream tools

---

## vLLM Alt Backend (blocked)

Preserved as `docker-compose.vllm.yml`. Currently fails on Gemma 4 12B due to upstream bug:

```
Resolved architecture: TransformersMultiModalForCausalLM
→ no vLLM implementation, falling back to Transformers implementation
→ Marlin GEMM: RuntimeError: Shape mismatch: a.size(1) = 4096, size_k = 8192
```

**Root cause**: vLLM has no native Gemma 4 implementation yet. Falls back to wrapping HF `transformers` model. Gemma 4 fused `gate_up_proj` layout doesn't match what vLLM Marlin W4A16 kernel expects (factor-of-2 K-dim mismatch).

**Will switch back when**: vLLM ships native Gemma 4 implementation. Track https://github.com/vllm-project/vllm/issues for `gemma 4` / `TransformersMultiModalForCausalLM marlin`.

### vLLM original intended config

| Item | Value |
|------|-------|
| Engine | vLLM (latest, currently v0.22+) |
| Checkpoint | `google/gemma-4-12B-it-qat-w4a16-ct` (compressed-tensors W4A16 QAT) |
| KV cache | FP8 storage (Ada upcast on read — no FP8 attention math on sm_89) |
| Kernels | Marlin W4A16 + FlashAttention v2 + FP8 KV |
| Scheduler | PagedAttention + continuous batching |
| Parsers | `--reasoning-parser gemma4`, `--tool-call-parser gemma4` |

vLLM theoretical advantage on this hardware:
- PagedAttention → better KV memory packing → 1.2-1.5× more concurrency at same ctx
- Native FP8 KV via Ada → ~10% extra KV budget vs Q8 GGUF
- Better tool/reasoning parser integration

When vLLM fixed, performance delta on L4 Gemma 12B vs current llama.cpp setup expected: ~+20-30% aggregate throughput, +10-20% concurrency, real tool/reasoning parsing built-in. Worth re-evaluating then.
