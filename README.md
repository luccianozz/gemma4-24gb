# Gemma 4 12B on L4 — Local Run

Personal-use Docker Compose setup. vLLM serves `google/gemma-4-12B-it-qat-w4a16-ct` on a single NVIDIA L4 (24GB).

Deep dives: [ANALYSIS.md](./ANALYSIS.md) (decision rationale), [ARCHITECTURE.md](./ARCHITECTURE.md) (system view).

## Prereqs

- NVIDIA GPU with 24GB+ VRAM (L4, A10G, RTX 4090, A100...)
- NVIDIA driver ≥525, CUDA ≥12.1
- Docker ≥24.x + Docker Compose v2
- NVIDIA Container Toolkit installed and configured
- HuggingFace account + Gemma 4 license accepted at https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct
- HF token (read scope) from https://huggingface.co/settings/tokens
- ~30GB free disk

Verify toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Run

```bash
cp .env.example .env          # paste HF_TOKEN
docker compose up -d
docker compose logs -f vllm   # watch ~5min cold start (weight download)
```

Wait for `Uvicorn running on http://0.0.0.0:8000`.

## Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Confirm FP8 KV cache actually enabled (silent fallback to FP16 = OOM later):
```bash
docker compose logs vllm | grep -i fp8
```

## Use

```bash
pip install -r requirements.txt
python main.py
python main.py "Write a haiku about quantization."
```

## Stop / restart

```bash
docker compose down              # stops; weights persist in hf_cache volume
docker compose down -v           # also drops volume (~7GB re-download next time)
docker compose restart vllm
```

## Troubleshoot

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `runtime: nvidia` error | Old Docker | Upgrade Docker ≥24, install nvidia-container-toolkit |
| 403 on weight download | Gemma license not accepted | Accept on HF model page (link above) |
| OOM at startup | Driver too old / wrong dtype | `nvidia-smi` → driver ≥525; verify FP8 enabled in logs |
| 401 on download | Invalid `HF_TOKEN` | Regenerate token (read scope) |
| Container restarts loop | Check logs | `docker compose logs vllm` |
| Slow first request | Weight load + JIT | Expected; warm requests fast |

## Files

- `docker-compose.yml` — vLLM service definition
- `main.py` — minimal OpenAI-SDK client
- `requirements.txt` — `openai` only (host-side)
- `.env.example` — HF token template
- `ANALYSIS.md` — why this setup
- `ARCHITECTURE.md` — VRAM math + kernel notes
- `PLAN.md` — full-repo plan (parked)
