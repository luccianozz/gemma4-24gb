# Gemma 4 12B Quantization — Deployment Analysis

## Problem

Run Gemma 4 12B (11.95B params) on a **g6.xlarge** EC2 instance with a single **NVIDIA L4 GPU (24GB VRAM)**.

- BF16 weights alone: ~24GB → no headroom for KV cache/activations → won't work
- Need quantization to fit model + inference overhead in 24GB

## Target Hardware

| Spec | Value |
|------|-------|
| Instance | g6.xlarge |
| GPU | NVIDIA L4 |
| Architecture | Ada Lovelace (sm_89) |
| VRAM | 24GB GDDR6 |
| INT8 throughput | 120 TOPS |
| INT4 throughput | 60 TOPS |

## Model Overview

| Property | Value |
|----------|-------|
| Parameters | 11.95B |
| Context window | 256K tokens |
| Modalities | Text, Image, Audio, Video |
| Base precision | BF16 (~24GB) |

## VRAM Budget by Quantization Level

| Component | BF16 | INT8 | INT4 (W4A16) |
|-----------|------|------|--------------|
| Weights | ~24GB | ~12GB | ~7GB |
| KV cache + activations | ~3-4GB | ~3-4GB | ~3-4GB |
| CUDA overhead | ~2GB | ~2GB | ~2GB |
| **Total** | **~29GB ❌** | **~17GB ✅** | **~12GB ✅** |
| **Free for context** | — | ~7GB | ~15GB |

## Available Pre-Quantized Checkpoints

### Official (Google)

| Checkpoint | Format | Use Case |
|------------|--------|----------|
| [gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) | HF (BF16 QAT weights) | Research, bitsandbytes loading |
| [gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) | GGUF Q4_0 | llama.cpp, Ollama |
| [gemma-4-12B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) | Compressed Tensors | vLLM |

All QAT checkpoints use **Quantization-Aware Training** — quantization simulated during training, so INT4 quality nearly matches BF16.

### Community

| Checkpoint | Format | Notes |
|------------|--------|-------|
| [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | GGUF (multiple quants) | Standard + QAT |
| [bartowski/gemma-4-12B-it-GGUF](https://huggingface.co/bartowski/gemma-4-12B-it-GGUF) | GGUF Q4_K_M (~7.4GB) | Popular community quant |
| [lmstudio-community/gemma-4-12B-it-GGUF](https://huggingface.co/lmstudio-community/gemma-4-12B-it-GGUF) | GGUF | LM Studio optimized |
| [AxionML/Gemma-4-12B-NVFP4](https://huggingface.co/AxionML/Gemma-4-12B-NVFP4) | NVIDIA FP4 | Requires Hopper GPU |

## Deployment Approaches Compared

### 1. vLLM + Compressed Tensors (`qat-w4a16-ct`)

Google published this checkpoint specifically for vLLM. Loads pre-quantized W4A16 weights directly. Uses optimized Marlin/CUTLASS CUDA kernels for INT4 matmul.

**Pros:**
- Production-grade inference server (OpenAI-compatible API)
- PagedAttention for optimal KV cache VRAM usage
- Continuous batching for concurrent requests
- L4 Ada GPU fully supported (sm_89)
- Tensor parallelism ready for multi-GPU scaling

**Cons:**
- Heavy dependency (vLLM is large)
- Slower cold start (~30-60s)
- Less flexibility for custom inference loops

**Best for:** Production API serving, multi-user workloads.

### 2. transformers + bitsandbytes (`qat-q4_0-unquantized`)

Load BF16 QAT checkpoint, bitsandbytes quantizes to NF4/INT4 at load time.

**Pros:**
- Pure Python, simple code
- Full control over generation
- Easy integration in Python pipelines
- Good for fine-tuning downstream

**Cons:**
- Slower kernels than vLLM (~1.5-2x)
- No continuous batching
- No PagedAttention
- Runtime quantization overhead at startup

**Best for:** Research, prototyping, single-user scripts.

### 3. llama.cpp / Ollama (`qat-q4_0-gguf`)

GGUF format loaded by llama.cpp C++ engine.

**Pros:**
- Simplest setup (Ollama = one command)
- Battle-tested, mature
- CPU offloading if VRAM tight

**Cons:**
- Not Python-native
- Less optimized on Ada than vLLM
- Multimodal support can lag behind HF

**Best for:** Local dev, quick testing.

### Comparison Matrix

```
                    Robustness  Performance  Simplicity  Flexibility
vLLM + CT           ★★★★★       ★★★★★        ★★★          ★★★
transformers + bnb  ★★★         ★★★          ★★★★         ★★★★★
llama.cpp/Ollama    ★★★★        ★★★★         ★★★★★        ★★
```

## Recommendation

**vLLM + `gemma-4-12B-it-qat-w4a16-ct`** is the most robust path.

### Why

1. Google published compressed tensors checkpoint specifically for this
2. PagedAttention = predictable VRAM, no OOM surprises under load
3. OpenAI-compatible API = swap models without changing client code
4. Concurrent requests, health checks, metrics built in
5. L4 Ada GPU well supported by Marlin INT4 kernels

### VRAM Projection on g6.xlarge

| Component | VRAM |
|-----------|------|
| Weights (W4A16) | ~7GB |
| CUDA overhead | ~2GB |
| Available for KV cache | ~15GB |
| **Realistic max context (FP8 KV)** | **~32K-48K tokens** |

Native model ceiling: 131072 tokens. L4 budget caps lower — measure before trusting.

## The 40GB Gap

The [official vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) lists **40GB+ GPU** as requirement for Gemma 4 12B. L4 = 24GB. Gap closed only by stacking three constraints:

1. **W4A16 QAT checkpoint** — `google/gemma-4-12B-it-qat-w4a16-ct`, not the BF16 base
2. **FP8 KV cache** — `--kv-cache-dtype fp8` cuts KV memory ~50%
3. **Capped `--max-model-len`** — 32K, not the 131K native ceiling

Drop any one → OOM under load. BF16 + full context will not fit.

Alternative: `google/gemma-4-E4B-it` (4B) runs unquantized on L4 with room to spare. Choose 12B only if quality delta justifies the engineering cost.

## vLLM Serve Command (L4-tuned)

```bash
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

Flag rationale:

| Flag | Why |
|------|-----|
| `--max-model-len 32768` | Caps KV cache footprint. Raise only after measuring headroom. |
| `--gpu-memory-utilization 0.92` | Leaves ~2GB for CUDA workspace on 24GB L4. |
| `--kv-cache-dtype fp8` | Mandatory for 12B on L4. ~50% KV cache reduction. |
| `--reasoning-parser gemma4` | Enables thinking-mode token extraction. |
| `--tool-call-parser gemma4` + `--enable-auto-tool-choice` | Function calling support. |
| `--mm-processor-kwargs` | Caps image token budget. Default = OOM risk. |
| `--limit-mm-per-prompt` | Hard ceiling on multimodal payload. |

### Architecture

```
Client → vLLM server (port 8000) → OpenAI-compatible API
              ↓
         gemma-4-12B-it-qat-w4a16-ct
              ↓
         NVIDIA L4 24GB (Marlin W4A16 + FP8 KV cache)
```

## References

- [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) — authoritative deployment guide
- [Google QAT checkpoints (HF)](https://huggingface.co/google) — official quantized weights