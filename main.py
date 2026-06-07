"""Minimal client for local Gemma 4 12B server (llama.cpp default, vLLM alt)."""

import os
import sys

from openai import OpenAI

MODEL = os.getenv("MODEL_NAME", "gemma-4-12b")

api_key = os.getenv("VLLM_API_KEY")
if not api_key:
    sys.exit("VLLM_API_KEY not set. export it or source .env first.")

client = OpenAI(
    base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"),
    api_key=api_key,
)

prompt = (
    sys.argv[1] if len(sys.argv) > 1 else "Explain INT4 quantization in two sentences."
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512,
    temperature=0.7,
)
print(resp.choices[0].message.content)
