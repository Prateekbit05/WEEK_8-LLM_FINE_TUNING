#!/usr/bin/env python3
"""
DAY 4 - Complete Inference Testing
Tests: Base model, Fine-tuned, INT8, INT4, GGUF
Measures: Tokens/sec, VRAM, Latency, Streaming, Batch
"""

import os
import sys
import time
import json
import subprocess
import torch
import psutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from threading import Thread
import warnings
warnings.filterwarnings('ignore')


# =============================================
# AUTO-INSTALL MISSING PACKAGES
# =============================================
def ensure_package(import_name: str, pip_name: str = None) -> bool:
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"   Installing {pip_name} ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"   ✅ {pip_name} installed")
            return True
        print(f"   ❌ Failed to install {pip_name}: {r.stderr.strip()[:100]}")
        return False

print("Checking dependencies ...")
ensure_package("accelerate", "accelerate")
ensure_package("peft",       "peft")
ensure_package("pandas",     "pandas")

# Optional — llama-cpp-python is large; skip silently if install fails
LLAMA_CPP_AVAILABLE = ensure_package("llama_cpp", "llama-cpp-python")
PEFT_AVAILABLE      = ensure_package("peft", "peft")

if PEFT_AVAILABLE:
    from peft import PeftModel, PeftConfig

if LLAMA_CPP_AVAILABLE:
    from llama_cpp import Llama

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto" if torch.cuda.is_available() else None  # device_map needs accelerate+CUDA

print(f"\nPEFT available      : {PEFT_AVAILABLE}")
print(f"llama.cpp available : {LLAMA_CPP_AVAILABLE}")
print(f"CUDA available      : {torch.cuda.is_available()}")
print(f"Running on          : {DEVICE.upper()}")


# =============================================
# CONFIGURATION
# =============================================
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH  = "../DAY_2_PEFT_LoRA_QLoRA/adapters"
GGUF_SEARCH_PATHS = [
    "../DAY_3_QUANTISATION/quantized/gguf/model.gguf",
    "../DAY_3_QUANTISATION/quantized/gguf/model-f16.gguf",
    "../DAY_3_QUANTISATION/quantized/gguf/model-q8_0.gguf",
    "../DAY_3_QUANTISATION/quantized/gguf/model-q4_0.gguf",
]

TEST_PROMPTS = [
    "What is machine learning?",
    "Explain artificial intelligence in simple terms.",
    "What are neural networks?",
]

BATCH_PROMPTS = [
    "What is Python?",
    "What is JavaScript?",
    "What is C++?",
    "What is Java?",
    "What is Rust?",
]

MAX_NEW_TOKENS = 100


# =============================================
# HELPERS
# =============================================
def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def cpu_mem_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)

def get_device(model):
    return next(model.parameters()).device

def load_model_cpu_safe(model_id_or_path, quant_cfg=None, dtype=torch.float16):
    """
    Load model correctly for both CPU-only and CUDA environments.
    - CUDA  : device_map='auto'  (requires accelerate)
    - CPU   : device_map=None, map to 'cpu' explicitly
    """
    kwargs = dict(trust_remote_code=True)

    if quant_cfg is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "INT8/INT4 quantization requires a CUDA GPU. "
                "No GPU detected — skipping quantized test."
            )
        kwargs["quantization_config"] = quant_cfg
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto"
    else:
        # CPU-only: load in fp32 (fp16 tensors on CPU cause issues on some builds)
        kwargs["torch_dtype"] = torch.float32

    return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)

def safe_tokenizer(path):
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def find_adapter_base_model():
    config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path", BASE_MODEL_ID)
        print(f"   Adapter base model: {base}")
        return base
    return BASE_MODEL_ID

def find_gguf():
    for p in GGUF_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    return None


# =============================================
# BENCHMARK FUNCTIONS
# =============================================
def measure_throughput(model, tokenizer, prompts):
    model.eval()
    device = get_device(model)
    total_tokens = 0
    total_time   = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        gen_tokens    = outputs.shape[1] - inputs['input_ids'].shape[1]
        total_tokens += gen_tokens
        total_time   += elapsed

    return total_tokens / total_time if total_time > 0 else 0


def measure_latency(model, tokenizer, prompt):
    model.eval()
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (time.time() - start) * 1000
    text       = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    return latency_ms, num_tokens, text


def test_streaming(model, tokenizer, prompt):
    model.eval()
    device = get_device(model)
    try:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs   = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs   = {k: v.to(device) for k, v in inputs.items()}

        kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        start  = time.time()
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        chunks = sum(1 for _ in streamer)
        thread.join()
        return time.time() - start, chunks
    except Exception as e:
        print(f"   ⚠️  Streaming failed: {e}")
        return 0.0, 0


def test_batch(model, tokenizer, prompts):
    model.eval()
    device = get_device(model)
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return time.time() - start, len(prompts)
    except Exception as e:
        print(f"   ⚠️  Batch failed: {e}")
        return 0.0, 0


def run_hf_test(model, tokenizer, name, fmt, vram_used):
    print("   Measuring throughput ...")
    tok_per_sec = measure_throughput(model, tokenizer, TEST_PROMPTS)

    print("   Measuring latency ...")
    latency, num_tok, text = measure_latency(model, tokenizer, TEST_PROMPTS[0])

    print("   Testing streaming ...")
    stream_time, chunks = test_streaming(model, tokenizer, TEST_PROMPTS[0])

    print("   Testing batch ...")
    batch_time, batch_size = test_batch(model, tokenizer, BATCH_PROMPTS)

    result = {
        'model_name':       name,
        'model_type':       fmt.lower().replace(' ', '_'),
        'format':           fmt,
        'tokens_per_sec':   round(tok_per_sec, 2),
        'vram_mb':          round(vram_used, 2),
        'latency_ms':       round(latency, 2),
        'num_tokens':       num_tok,
        'streaming_time':   round(stream_time, 2),
        'streaming_chunks': chunks,
        'batch_time':       round(batch_time, 2),
        'batch_size':       batch_size,
        'sample_output':    text[:300] if text else "",
    }

    print(f"\n   ✅ {name}:")
    print(f"      Throughput : {tok_per_sec:.2f} tok/s")
    print(f"      VRAM       : {vram_used:.2f} MB")
    print(f"      Latency    : {latency:.2f} ms")
    print(f"      Streaming  : {stream_time:.2f}s ({chunks} chunks)")
    print(f"      Batch ({batch_size})  : {batch_time:.2f}s")
    return result


# =============================================
# TEST FUNCTIONS
# =============================================
def test_1_base_model():
    print("\n" + "="*70)
    print("TEST 1: Base Model (FP16)")
    print("="*70)
    clear_mem()
    mem_before = gpu_mem_mb()
    try:
        tok = safe_tokenizer(BASE_MODEL_ID)
        model = load_model_cpu_safe(BASE_MODEL_ID)
        print("   ✅ Model loaded")

        vram   = gpu_mem_mb() - mem_before
        result = run_hf_test(model, tok, "Base Model (FP16)", "FP16", vram)
        del model, tok
        clear_mem()
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_2_finetuned():
    print("\n" + "="*70)
    print("TEST 2: Fine-tuned Model")
    print("="*70)

    if not os.path.exists(ADAPTER_PATH):
        print(f"   ⚠️  Adapters not found at {ADAPTER_PATH} — skipping")
        return None
    if not PEFT_AVAILABLE:
        print("   ⚠️  PEFT not installed — skipping")
        return None

    clear_mem()
    mem_before = gpu_mem_mb()
    try:
        adapter_base = find_adapter_base_model()
        tok   = safe_tokenizer(adapter_base)
        model = load_model_cpu_safe(adapter_base)

        print("   Loading adapters ...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("   ✅ Adapters merged")

        vram   = gpu_mem_mb() - mem_before
        result = run_hf_test(model, tok, "Fine-tuned (LoRA)", "FP16+LoRA", vram)
        del model, tok
        clear_mem()
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_3_int8():
    print("\n" + "="*70)
    print("TEST 3: Quantized (INT8)")
    print("="*70)

    if not torch.cuda.is_available():
        print("   ⚠️  INT8 requires a CUDA GPU — skipping (running on CPU)")
        return None

    clear_mem()
    mem_before = gpu_mem_mb()
    try:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        tok = safe_tokenizer(BASE_MODEL_ID)
        model = load_model_cpu_safe(BASE_MODEL_ID, quant_cfg=cfg)
        print("   ✅ INT8 model loaded")

        vram   = gpu_mem_mb() - mem_before
        result = run_hf_test(model, tok, "Quantized INT8", "INT8", vram)
        del model, tok
        clear_mem()
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_4_int4():
    print("\n" + "="*70)
    print("TEST 4: Quantized (INT4/NF4)")
    print("="*70)

    if not torch.cuda.is_available():
        print("   ⚠️  INT4 requires a CUDA GPU — skipping (running on CPU)")
        return None

    clear_mem()
    mem_before = gpu_mem_mb()
    try:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tok   = safe_tokenizer(BASE_MODEL_ID)
        model = load_model_cpu_safe(BASE_MODEL_ID, quant_cfg=cfg)
        print("   ✅ INT4 model loaded")

        vram   = gpu_mem_mb() - mem_before
        result = run_hf_test(model, tok, "Quantized INT4/NF4", "INT4/NF4", vram)
        del model, tok
        clear_mem()
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_5_gguf():
    print("\n" + "="*70)
    print("TEST 5: GGUF (llama.cpp)")
    print("="*70)

    if not LLAMA_CPP_AVAILABLE:
        print("   ⚠️  llama-cpp-python not installed — skipping")
        print("   Install manually: pip install llama-cpp-python")
        return None

    gguf_path = find_gguf()
    if not gguf_path:
        print("   ⚠️  No GGUF file found in search paths — skipping")
        return None

    print(f"   Using: {gguf_path}")
    clear_mem()
    cpu_before = cpu_mem_mb()

    try:
        llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("   ✅ GGUF model loaded")
        ram_used = cpu_mem_mb() - cpu_before

        # Throughput
        print("   Measuring throughput ...")
        total_tok = 0; total_time = 0
        for prompt in TEST_PROMPTS:
            start = time.time()
            out   = llm(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.7, echo=False)
            elapsed    = time.time() - start
            total_tok += len(out['choices'][0]['text'].split())
            total_time += elapsed
        tok_per_sec = total_tok / total_time if total_time > 0 else 0

        # Latency
        print("   Measuring latency ...")
        start   = time.time()
        out     = llm(TEST_PROMPTS[0], max_tokens=MAX_NEW_TOKENS, temperature=0.7, echo=False)
        latency = (time.time() - start) * 1000
        text    = out['choices'][0]['text']
        num_tok = len(text.split())

        # Streaming
        print("   Testing streaming ...")
        start  = time.time()
        chunks = sum(1 for _ in llm(TEST_PROMPTS[0], max_tokens=MAX_NEW_TOKENS, stream=True))
        stream_time = time.time() - start

        # Batch (sequential in llama.cpp)
        print("   Testing batch ...")
        start = time.time()
        for p in BATCH_PROMPTS:
            llm(p, max_tokens=50, echo=False)
        batch_time = time.time() - start

        result = {
            'model_name':       'GGUF (llama.cpp)',
            'model_type':       'gguf',
            'format':           'GGUF',
            'tokens_per_sec':   round(tok_per_sec, 2),
            'vram_mb':          0,
            'ram_mb':           round(ram_used, 2),
            'latency_ms':       round(latency, 2),
            'num_tokens':       num_tok,
            'streaming_time':   round(stream_time, 2),
            'streaming_chunks': chunks,
            'batch_time':       round(batch_time, 2),
            'batch_size':       len(BATCH_PROMPTS),
            'sample_output':    text[:300] if text else "",
        }

        print(f"\n   ✅ GGUF:")
        print(f"      Throughput : {tok_per_sec:.2f} tok/s")
        print(f"      RAM        : {ram_used:.2f} MB")
        print(f"      Latency    : {latency:.2f} ms")

        del llm
        clear_mem()
        return result

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("DAY 4 — INFERENCE OPTIMIZATION & BENCHMARKING")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_total:.1f} GB")
    else:
        ram_total = psutil.virtual_memory().total / 1e9
        print(f"RAM : {ram_total:.1f} GB (CPU-only mode)")
    print("="*70)

    results = []
    for test_fn in [test_1_base_model, test_2_finetuned, test_3_int8, test_4_int4, test_5_gguf]:
        r = test_fn()
        if r:
            results.append(r)

    if not results:
        print("\n❌ No tests completed successfully.")
        return

    import pandas as pd
    os.makedirs('benchmarks', exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv('benchmarks/results.csv', index=False)
    print(f"\n✅ Saved: benchmarks/results.csv")

    with open('benchmarks/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved: benchmarks/results.json")

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    cols  = ['model_name', 'format', 'tokens_per_sec', 'vram_mb', 'latency_ms']
    avail = [c for c in cols if c in df.columns]
    print("\n" + df[avail].to_string(index=False))

    if df['tokens_per_sec'].notna().any():
        fastest     = df.loc[df['tokens_per_sec'].idxmax()]
        lowest_lat  = df.loc[df['latency_ms'].idxmin()]
        print(f"\n🏆 Fastest       : {fastest['model_name']} ({fastest['tokens_per_sec']:.2f} tok/s)")
        print(f"🏆 Lowest Latency: {lowest_lat['model_name']} ({lowest_lat['latency_ms']:.2f} ms)")

    print("\n" + "="*70)
    print("✅ ALL BENCHMARKS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()