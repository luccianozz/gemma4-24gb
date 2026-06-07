# Architecture — Gemma 4 12B on L4

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Developer laptop                                                │
│   └── SSH / SSM                                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  AWS EC2 g6.xlarge  (Ubuntu 22.04, DL AMI)                       │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────┐     │
│  │ tmux: vllm serve    │    │ Python venv (.venv)          │     │
│  │  ├─ port 8000 (API) │◄───┤  ├─ main.py (OpenAI client)  │     │
│  │  ├─ /v1/chat        │    │  └─ benchmark.py             │     │
│  │  ├─ /v1/completions │    └──────────────────────────────┘     │
│  │  ├─ /v1/models      │                                         │
│  │  └─ /health         │                                         │
│  └──────────┬──────────┘                                         │
│             │                                                    │
│             ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ NVIDIA L4 (24GB GDDR6, Ada Lovelace sm_89)              │     │
│  │  ├─ Weights: W4A16 QAT      ~7GB                        │     │
│  │  ├─ KV cache: FP8           ~13GB (32K ctx, batched)    │     │
│  │  ├─ Activations             ~2GB                        │     │
│  │  └─ CUDA workspace          ~2GB                        │     │
│  │                                                         │     │
│  │ Kernels (Ada sm_89):                                    │     │
│  │   ├─ Marlin W4A16 matmul (INT8 tensor cores)            │     │
│  │   ├─ FlashAttention v2 (BF16 compute)                   │     │
│  │   └─ FP8 KV cache (storage only, upcast on read)        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Disk: 80GB gp3                                                  │
│   ├─ ~/.cache/huggingface/  (~7GB weights + metadata)            │
│   ├─ .venv/                 (~6GB torch + vLLM)                  │
│   └─ vllm.log, benchmark_results.csv                             │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### Model layer

| Item | Value |
|------|-------|
| Checkpoint | `google/gemma-4-12B-it-qat-w4a16-ct` |
| Format | Compressed Tensors (HF) |
| Quantization | W4A16 QAT (4-bit weights, 16-bit activations, trained-in) |
| KV cache dtype | FP8 (runtime, via `--kv-cache-dtype fp8`) |
| Max context | 32768 tokens (capped; native 131072) |
| Modalities | Text + image + audio (capped via mm flags) |

### Runtime layer

| Item | Value |
|------|-------|
| Engine | vLLM ≥0.8.5 |
| Matmul kernel | Marlin W4A16 (uses INT8 tensor cores on sm_89) |
| Attention | FlashAttention v2 + BF16 compute (FA v3 / FP8 attention = Hopper-only) |
| KV cache | FP8 storage only — upcast to BF16 on attention read (no FP8 tensor cores on Ada) |
| Scheduler | PagedAttention + continuous batching |
| API | OpenAI-compatible REST, port 8000 |
| Process mgr | tmux session |
| Parsers | `--reasoning-parser gemma4`, `--tool-call-parser gemma4` |

### Infra layer

| Item | Value |
|------|-------|
| Instance | AWS g6.xlarge |
| GPU | NVIDIA L4, 24GB, sm_89 |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04) |
| Driver | ≥525 |
| CUDA | ≥12.1 |
| Disk | 80GB gp3 |
| Network | Default VPC, SG: 22 inbound only |

## Request Flow

```
1. Client → POST /v1/chat/completions (OpenAI SDK)
2. vLLM scheduler:
   ├─ tokenize prompt
   ├─ allocate KV cache pages (PagedAttention)
   ├─ batch with concurrent requests (continuous batching)
   └─ schedule on GPU
3. Forward pass:
   ├─ Marlin W4A16 matmul (weights stay INT4 in VRAM)
   ├─ Attention reads FP8 KV cache pages
   └─ Sample next token
4. Stream chunk → client
5. Loop until EOS or max_tokens
6. Free KV pages, return final usage stats
```

## VRAM Budget @ 32K context

L4 hardware total = 24GB. `--gpu-memory-utilization 0.92` caps vLLM at **~22GB**. Remaining ~2GB held by NVIDIA driver + CUDA runtime outside vLLM. **22GB is the real budget.**

| Component | VRAM | Notes |
|-----------|------|-------|
| NVIDIA driver + CUDA runtime | ~2 GB | Outside vLLM (reserved by `--gpu-memory-utilization 0.92`) |
| **vLLM budget (0.92 × 24GB)** | **~22 GB** | |
| └─ Model weights (W4A16) | ~7 GB | Static, loaded once |
| └─ Activations | ~2 GB | Transient per step |
| └─ vLLM workspace | ~1 GB | Scheduler, paged tables |
| └─ KV cache (FP8) available | **~12 GB** | Scales w/ ctx × concurrent |
| **Hardware total** | **24 GB** | |

KV cache @ FP8 ≈ ~80 KB/token for 12B-class model. 32K tokens × 80 KB ≈ **2.5 GB per request**. 12 GB KV budget ÷ 2.5 GB ≈ **~4-5 concurrent requests at full 32K context**. Measure to confirm.

Trade-offs:
- Drop ctx to 16K → ~1.3 GB/request → ~8-10 concurrent at full ctx
- Drop FP8 KV → KV doubles to ~5 GB/request → ~2 concurrent at 32K
- Drop W4A16 (use BF16) → ~24GB weights alone → OOM before any KV
- Raise `--gpu-memory-utilization` past 0.92 → high OOM risk from driver/CUDA workspace squeeze

## Failure Modes

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| OOM at startup | vLLM crash before `/health` ready | Wrong dtype, missing flag | Verify W4A16 + FP8 + max-len=32K all set |
| Silent FP8 fallback | KV uses FP16, OOM at half expected ctx | Wrong CUDA build | Grep log: `"FP8 KV cache enabled"` |
| 403 on download | HF fetch fails | License not accepted | Visit model page, accept Google terms |
| Tool calls return raw text | Parser disabled | Old vLLM version | Upgrade ≥0.8.5, verify `--tool-call-parser gemma4` |
| Hang under load | Request times out, no 500 | KV exhaustion mid-request | Lower concurrent limit, lower max-model-len |
| Multimodal OOM | Image request crashes | No mm caps | Both `--mm-processor-kwargs` + `--limit-mm-per-prompt` set |

## Scaling Path (out of scope, documented)

| Need | Path |
|------|------|
| More throughput | g6.12xlarge (4× L4) + `--tensor-parallel-size 4` |
| More context (>32K) | g5.12xlarge or g6.12xlarge — single L4 KV cache caps here |
| Higher quality | Drop to `google/gemma-4-12B-it` BF16 on A10G/L40S 48GB |
| Lower latency | Sticky-session client, prefix caching, speculative decoding |

## Security Notes

- `HF_TOKEN` only in `.env` (gitignored). Never in code, never in serve.sh
- vLLM API has no auth by default. Keep port 8000 localhost-only OR put behind reverse proxy + API key
- Multimodal: image URLs fetched by vLLM — SSRF risk if client-controlled URLs allowed. Validate or proxy
- Model output: not sanitized. Tool call args from model = untrusted input downstream