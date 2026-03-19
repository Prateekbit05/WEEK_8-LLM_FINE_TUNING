#!/usr/bin/env python3
"""
evaluate_models.py
Evaluates quantized models across formats: fp16, int8, int4, gguf.

Usage:
    python scripts/evaluate_models.py --model-dir quantized --formats fp16 int8 int4 gguf

Metrics collected per format:
    - Model size on disk (MB)
    - Load time (s)
    - First-token latency (s)
    - Avg tokens/sec throughput
    - Peak GPU memory (MB)
    - Perplexity on eval prompts
    - Sample generation output
"""

import os
import gc
import sys
import json
import time
import math
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

# Default prompts used for perplexity + throughput evaluation
EVAL_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is gradient descent and how does it work?",
    "Describe the transformer architecture in simple terms.",
    "What are the main advantages of quantization in deep learning?",
    "How does attention mechanism work in neural networks?",
]

GENERATION_PROMPT = "Explain what a large language model is in two sentences:"

FORMAT_DIRS = {
    "fp16": "model-fp16",
    "int8": "model-int8",
    "int4": "model-int4",
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def banner(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 ** 2)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)


def clear_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quantized model formats.")
    parser.add_argument("--model-dir",   type=str, default="quantized",
                        help="Root directory containing model-fp16/, model-int8/, etc.")
    parser.add_argument("--formats",     nargs="+",
                        choices=["fp16", "int8", "int4", "gguf"],
                        default=["fp16", "int8", "int4"],
                        help="Formats to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Tokens to generate per prompt during throughput test")
    parser.add_argument("--llama-cpp-dir", type=str, default="llama.cpp",
                        help="Path to llama.cpp repo (needed for GGUF eval)")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Where to save results (default: --model-dir)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
    fmt: str,
) -> tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer], float]:
    """
    Load model + tokenizer for a given format.
    Returns (model, tokenizer, load_time_seconds).
    """
    reset_gpu_stats()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if fmt == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    elif fmt == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    elif fmt == "int4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        raise ValueError(f"Unsupported format for HF loading: {fmt}")

    model.eval()
    load_time = time.time() - t0
    return model, tokenizer, load_time


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def measure_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
) -> tuple[float, float, int]:
    """
    Returns (first_token_latency_s, total_time_s, tokens_generated).
    Uses a two-phase approach: generate 1 token for latency, then the rest.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # First-token latency
    with torch.no_grad():
        t0 = time.time()
        _ = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        first_token_latency = time.time() - t0

    # Full generation
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        total_time = time.time() - t0

    tokens_generated = out.shape[1] - inputs["input_ids"].shape[1]
    return first_token_latency, total_time, tokens_generated


def measure_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
) -> float:
    """
    Compute average perplexity across a list of prompts.
    Lower = better.
    """
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss  # mean cross-entropy over tokens

        n_tokens = input_ids.shape[1]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def generate_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Generate a text sample for qualitative inspection."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


# ──────────────────────────────────────────────
# GGUF evaluation via llama.cpp CLI
# ──────────────────────────────────────────────

def evaluate_gguf(
    gguf_path: str,
    llama_cpp_dir: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """
    Run GGUF inference using llama.cpp's llama-cli or main binary.
    Returns a metrics dict.
    """
    llama_cpp = Path(llama_cpp_dir)

    # Find the CLI binary (name varies by llama.cpp version)
    binary = None
    for candidate in ["llama-cli", "main", "llama-run"]:
        p = llama_cpp / candidate
        if p.exists():
            binary = p
            break
        p = llama_cpp / "build" / "bin" / candidate
        if p.exists():
            binary = p
            break

    if binary is None:
        return {
            "error": (
                "llama.cpp binary not found. Build it first:\n"
                "  cd llama.cpp && cmake -B build && cmake --build build --config Release"
            )
        }

    size_mb = dir_size_mb(gguf_path)
    cmd = [
        str(binary),
        "-m", gguf_path,
        "-p", prompt,
        "-n", str(max_new_tokens),
        "--log-disable",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:300], "size_mb": round(size_mb, 2)}

    output_text = result.stdout.strip()
    # Rough token estimate (llama.cpp doesn't expose exact counts via stdout easily)
    rough_tokens = len(output_text.split())
    tokens_per_sec = rough_tokens / elapsed if elapsed > 0 else 0

    return {
        "size_mb": round(size_mb, 2),
        "load_time_s": None,          # not separately measurable via CLI
        "first_token_latency_s": None,
        "total_generation_time_s": round(elapsed, 3),
        "tokens_generated": rough_tokens,
        "tokens_per_second": round(tokens_per_sec, 2),
        "peak_gpu_mb": None,          # not available via CLI
        "perplexity": None,           # not available via CLI
        "sample_output": output_text[:300],
    }


# ──────────────────────────────────────────────
# Per-format evaluation
# ──────────────────────────────────────────────

def evaluate_hf_format(
    model_path: str,
    fmt: str,
    max_new_tokens: int,
) -> dict:
    """Full evaluation for fp16 / int8 / int4."""
    size_mb = dir_size_mb(model_path)
    metrics: dict = {"size_mb": round(size_mb, 2), "format": fmt}

    print(f"  Loading model from {model_path} ...")
    try:
        model, tokenizer, load_time = load_model_and_tokenizer(model_path, fmt)
    except Exception as exc:
        print(f"  ❌ Load failed: {exc}")
        return {"error": str(exc), "size_mb": round(size_mb, 2)}

    metrics["load_time_s"] = round(load_time, 3)
    print(f"  ✅ Loaded in {load_time:.2f}s")

    # ── Latency + throughput ───────────────────────────────────────────────────
    print(f"  Measuring latency / throughput ...")
    try:
        first_lat, total_time, n_tokens = measure_latency(
            model, tokenizer, GENERATION_PROMPT, max_new_tokens
        )
        tokens_per_sec = n_tokens / total_time if total_time > 0 else 0
        metrics["first_token_latency_s"]    = round(first_lat, 4)
        metrics["total_generation_time_s"]  = round(total_time, 3)
        metrics["tokens_generated"]         = n_tokens
        metrics["tokens_per_second"]        = round(tokens_per_sec, 2)
        print(f"  ✅ {tokens_per_sec:.1f} tok/s  |  first-token: {first_lat*1000:.1f}ms")
    except Exception as exc:
        print(f"  ⚠️  Latency measurement failed: {exc}")
        metrics["latency_error"] = str(exc)

    # ── Peak GPU memory ────────────────────────────────────────────────────────
    metrics["peak_gpu_mb"] = round(peak_gpu_mb(), 1)

    # ── Perplexity ─────────────────────────────────────────────────────────────
    print(f"  Computing perplexity on {len(EVAL_PROMPTS)} prompts ...")
    try:
        ppl = measure_perplexity(model, tokenizer, EVAL_PROMPTS)
        metrics["perplexity"] = round(ppl, 4)
        print(f"  ✅ Perplexity: {ppl:.4f}")
    except Exception as exc:
        print(f"  ⚠️  Perplexity failed: {exc}")
        metrics["perplexity_error"] = str(exc)

    # ── Sample generation ──────────────────────────────────────────────────────
    print(f"  Generating sample output ...")
    try:
        sample = generate_sample(model, tokenizer, GENERATION_PROMPT, max_new_tokens)
        metrics["sample_output"] = sample
        print(f"  ✅ Sample: {sample[:80]}...")
    except Exception as exc:
        print(f"  ⚠️  Sample generation failed: {exc}")
        metrics["sample_error"] = str(exc)

    del model, tokenizer
    clear_mem()
    return metrics


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    banner("EVALUATION RESULTS")

    header = f"{'Format':<8} {'Size(MB)':<12} {'Load(s)':<10} {'1st-tok(ms)':<14} {'Tok/s':<10} {'GPU(MB)':<10} {'Perplexity':<12}"
    print(header)
    print("-" * len(header))

    for fmt, m in all_results.items():
        if "error" in m and len(m) <= 2:
            print(f"{fmt:<8} ❌ {m.get('error','unknown')[:50]}")
            continue

        size    = f"{m.get('size_mb', 'N/A')}"
        load    = f"{m.get('load_time_s', 'N/A')}"
        lat_ms  = f"{m['first_token_latency_s']*1000:.1f}" if m.get('first_token_latency_s') else "N/A"
        tps     = f"{m.get('tokens_per_second', 'N/A')}"
        gpu     = f"{m.get('peak_gpu_mb', 'N/A')}"
        ppl     = f"{m.get('perplexity', 'N/A')}"
        print(f"{fmt:<8} {size:<12} {load:<10} {lat_ms:<14} {tps:<10} {gpu:<10} {ppl:<12}")


def write_markdown_report(all_results: dict, output_path: Path) -> None:
    rows = ""
    for fmt, m in all_results.items():
        if "error" in m and len(m) <= 2:
            rows += f"| {fmt} | ❌ Failed | — | — | — | — | — |\n"
            continue
        size    = m.get("size_mb", "N/A")
        load    = m.get("load_time_s", "N/A")
        lat_ms  = f"{m['first_token_latency_s']*1000:.1f} ms" if m.get("first_token_latency_s") else "N/A"
        tps     = m.get("tokens_per_second", "N/A")
        gpu     = f"{m.get('peak_gpu_mb','N/A')} MB"
        ppl     = m.get("perplexity", "N/A")
        rows += f"| {fmt} | {size} MB | {load}s | {lat_ms} | {tps} | {gpu} | {ppl} |\n"

    samples = ""
    for fmt, m in all_results.items():
        if m.get("sample_output"):
            samples += f"\n### {fmt.upper()}\n```\n{m['sample_output'][:500]}\n```\n"

    report = f"""# Model Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Metrics Comparison

| Format | Size | Load Time | 1st Token | Tok/s | GPU Mem | Perplexity |
|--------|------|-----------|-----------|-------|---------|------------|
{rows}
> **Perplexity**: lower is better. **Tok/s**: higher is better. **1st Token**: lower is better.

## Sample Outputs

*Prompt: "{GENERATION_PROMPT}"*
{samples}

## Notes

- Perplexity measured on {len(EVAL_PROMPTS)} evaluation prompts
- GPU memory = peak allocated during generation (CUDA only)
- GGUF throughput is estimated from output word count (llama.cpp CLI)
- INT8 requires `bitsandbytes>=0.41`; INT4 requires `bitsandbytes>=0.39`

---
*Generated by evaluate_models.py*
"""

    output_path.write_text(report)
    print(f"\n✅ Markdown report saved → {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for fmt in args.formats:
        banner(f"Evaluating: {fmt.upper()}")

        if fmt == "gguf":
            gguf_path = str(Path(args.model_dir) / "gguf" / "model.gguf")
            if not Path(gguf_path).exists():
                print(f"  ⚠️  GGUF file not found: {gguf_path} — skipping.")
                all_results[fmt] = {"error": f"File not found: {gguf_path}"}
                continue
            all_results[fmt] = evaluate_gguf(
                gguf_path, args.llama_cpp_dir, GENERATION_PROMPT, args.max_new_tokens
            )
        else:
            model_path = str(Path(args.model_dir) / FORMAT_DIRS[fmt])
            if not Path(model_path).exists():
                print(f"  ⚠️  Model dir not found: {model_path} — skipping.")
                all_results[fmt] = {"error": f"Directory not found: {model_path}"}
                continue
            all_results[fmt] = evaluate_hf_format(model_path, fmt, args.max_new_tokens)

        all_results[fmt]["format"] = fmt

    # ── Print table ────────────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = output_dir / "evaluation_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"✅ JSON results saved → {json_path}")

    # ── Save markdown report ───────────────────────────────────────────────────
    write_markdown_report(all_results, output_dir / "EVALUATION-REPORT.md")


if __name__ == "__main__":
    main()