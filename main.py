"""Minimal client for local vLLM Gemma 4 12B server."""

import os
import sys

from openai import OpenAI

MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"

client = OpenAI(
    base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"),
    api_key="EMPTY",
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
