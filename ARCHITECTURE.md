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

### Concurrency / context trade-off matrix

| Slots | Per-slot ctx | Per-user tok/s (est) | Aggregate tok/s | Best for |
|-------|--------------|---------------------|-----------------|----------|
| 4 | 32K | ~25-35 | ~100-140 | Long-context, low-load |
| **8** | **16K** | **~15-25** | **~120-200** | **Medium chat (balanced)** |
| **16** | **8K** | **~7-12** | **~110-190** | **Default — short-chat fleet** |
| 32 | 4K | ~3-6 | ~95-190 | Quick Q&A, max users |
| 64 | 2K | ~1.5-3 | ~95-190 | Borderline useful |

Hard ceiling: aggregate tok/s plateaus around 150-200 because L4 compute saturates. More slots = more users, lower per-user.

**300 concurrent on single L4 = not viable.** Sub-1 tok/s/user. For 300+ concurrent: 2× A100 80GB or 1× H100 (see Scaling Path).

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
