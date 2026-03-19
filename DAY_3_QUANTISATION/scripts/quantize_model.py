#!/usr/bin/env python3
"""
DAY 3: Model Quantization Pipeline
Supports FP16, INT8, INT4/NF4, and GGUF formats.
"""

import os
import gc
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a HuggingFace model to multiple formats.")
    parser.add_argument("--base-model",    type=str, required=True,  help="HF model ID or local path")
    parser.add_argument("--adapter-path",  type=str, default=None,   help="LoRA/PEFT adapter path (optional)")
    parser.add_argument("--output-dir",    type=str, default="quantized", help="Root output directory")
    parser.add_argument("--llama-cpp-dir", type=str, default="llama.cpp",  help="Path to llama.cpp repo")
    parser.add_argument("--skip-int8",     action="store_true",       help="Skip INT8 quantization")
    parser.add_argument("--skip-int4",     action="store_true",       help="Skip INT4 quantization")
    parser.add_argument("--skip-gguf",     action="store_true",       help="Skip GGUF conversion")
    return parser.parse_args()


def dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024 ** 2)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)


def clear_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def banner(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def ensure_package(package: str, pip_name: str = None) -> bool:
    """Try to import a package; install it if missing."""
    pip_name = pip_name or package
    try:
        __import__(package)
        return True
    except ImportError:
        print(f"  Installing missing dependency: {pip_name} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {pip_name} installed")
            return True
        else:
            print(f"  ❌ Failed to install {pip_name}: {result.stderr.strip()}")
            return False


def check_adapter_compatibility(base_model_name: str, adapter_path: str) -> tuple[bool, str]:
    """
    Read the adapter_config.json and check whether the base_model_name_or_path
    inside it matches the model we are loading. Returns (compatible, message).
    """
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        return False, f"adapter_config.json not found in {adapter_path}"

    with open(config_file) as f:
        adapter_cfg = json.load(f)

    adapter_base = adapter_cfg.get("base_model_name_or_path", "")

    # Normalise: strip trailing slashes, compare last path component or full string
    def norm(s):
        return s.rstrip("/").lower()

    if norm(adapter_base) == norm(base_model_name):
        return True, f"Adapter base matches: {adapter_base}"

    # Check hidden-size match via PeftConfig to catch obvious architecture mismatches
    msg = (
        f"⚠️  Adapter was trained on '{adapter_base}' "
        f"but you are loading '{base_model_name}'.\n"
        f"   These are DIFFERENT models — adapter weights will NOT be merged.\n"
        f"   The pipeline will continue with the base model only (no adapter).\n"
        f"   To fix: retrain the adapter on '{base_model_name}' or "
        f"pass the correct --base-model."
    )
    return False, msg


# ──────────────────────────────────────────────
# Quantization steps
# ──────────────────────────────────────────────

def save_fp16(args) -> tuple[str, float, bool]:
    """
    Load base model (+ optional adapters) and save as FP16.
    Returns (output_path, size_mb, adapter_merged).
    """
    banner("STEP 1: FP16 Baseline")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model in FP16: {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_merged = False

    if args.adapter_path and os.path.exists(args.adapter_path):
        # ── Compatibility check BEFORE attempting merge ────────────────────────
        compatible, compat_msg = check_adapter_compatibility(args.base_model, args.adapter_path)
        print(compat_msg)

        if not compatible:
            print("   Skipping adapter merge — saving base model only.")
        else:
            print(f"Merging adapters from {args.adapter_path} ...")
            try:
                model = PeftModel.from_pretrained(model, args.adapter_path)
                model = model.merge_and_unload()
                adapter_merged = True
                print("✅ Adapters merged successfully")
            except Exception as exc:
                print(f"❌ Adapter merge failed: {exc}")
                print("   Saving base model only — continuing pipeline.")
    elif args.adapter_path:
        print(f"⚠️  Adapter path not found: {args.adapter_path} — skipping.")

    out = f"{args.output_dir}/model-fp16"
    print(f"Saving to {out} ...")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    size = dir_size_mb(out)
    print(f"✅ FP16 saved — {size:.1f} MB  (adapter merged: {adapter_merged})")

    del model
    clear_mem()
    return out, size, adapter_merged


def quantize_int8(fp16_path: str, tokenizer, output_dir: str, fp16_mb: float) -> dict:
    """BitsAndBytes INT8 quantization."""
    banner("STEP 2: INT8 Quantization")
    try:
        cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            fp16_path,
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        out = f"{output_dir}/model-int8"
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        size = dir_size_mb(out)
        del model; clear_mem()

        ratio = fp16_mb / size
        print(f"✅ INT8 — {size:.1f} MB  ({ratio:.2f}× smaller)")
        return {"size_mb": round(size, 2), "compression": round(ratio, 2)}

    except Exception as exc:
        print(f"❌ INT8 failed: {exc}")
        return {"error": str(exc)}


def quantize_int4(fp16_path: str, tokenizer, output_dir: str, fp16_mb: float) -> dict:
    """BitsAndBytes NF4 (INT4) quantization with double quantization."""
    banner("STEP 3: INT4 / NF4 Quantization")
    try:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            fp16_path,
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        out = f"{output_dir}/model-int4"
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        size = dir_size_mb(out)
        del model; clear_mem()

        ratio = fp16_mb / size
        print(f"✅ INT4 — {size:.1f} MB  ({ratio:.2f}× smaller)")
        return {"size_mb": round(size, 2), "compression": round(ratio, 2)}

    except Exception as exc:
        print(f"❌ INT4 failed: {exc}")
        return {"error": str(exc)}


def convert_gguf(fp16_path: str, output_dir: str, llama_cpp_dir: str, fp16_mb: float) -> dict:
    """Convert FP16 model to GGUF via llama.cpp."""
    banner("STEP 4: GGUF Conversion")

    # ── Ensure the saved FP16 is a clean (non-PEFT) model ─────────────────────
    # If LoRA artifacts are present in safetensors, GGUF conversion will fail
    # with "Can not map tensor 'model.layers.X.self_attn.q_proj.base_layer.bias'"
    safetensors_file = Path(fp16_path) / "model.safetensors"
    if safetensors_file.exists():
        try:
            from safetensors import safe_open
            with safe_open(str(safetensors_file), framework="pt") as f:
                keys = list(f.keys())
            peft_keys = [k for k in keys if "base_layer" in k or "lora_A" in k or "lora_B" in k]
            if peft_keys:
                print(f"❌ GGUF skipped — FP16 model still contains {len(peft_keys)} LoRA/PEFT "
                      f"tensor(s) (e.g. '{peft_keys[0]}').")
                print("   This means the adapter merge failed in Step 1 (architecture mismatch).")
                print("   Fix: retrain the adapter on the correct base model, then re-run.")
                return {"error": "FP16 model contains unmerged PEFT weights — GGUF conversion requires a clean model"}
        except ImportError:
            pass  # safetensors not available for inspection; proceed and let llama.cpp fail naturally

    # ── Pre-flight: ensure sentencepiece + gguf are installed ─────────────────
    print("Checking GGUF dependencies ...")
    missing_deps = []
    for pkg, pip_pkg in [("sentencepiece", "sentencepiece"), ("gguf", "gguf")]:
        if not ensure_package(pkg, pip_pkg):
            missing_deps.append(pip_pkg)

    if missing_deps:
        msg = f"Required packages could not be installed: {', '.join(missing_deps)}"
        print(f"❌ {msg}\n   Fix manually: pip install {' '.join(missing_deps)}")
        return {"error": msg}

    # ── Locate convert script ──────────────────────────────────────────────────
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = Path(llama_cpp_dir) / "convert.py"

    if not convert_script.exists():
        msg = f"llama.cpp not found at '{llama_cpp_dir}'. Clone it and pass --llama-cpp-dir."
        print(f"⚠️  {msg}")
        return {"error": msg}

    gguf_dir = Path(output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_out = gguf_dir / "model.gguf"

    cmd = [
        sys.executable, str(convert_script),
        fp16_path,
        "--outfile", str(gguf_out),
        "--outtype", "f16",
    ]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not gguf_out.exists():
        print(f"❌ GGUF conversion failed:\n{result.stderr}")
        return {"error": result.stderr.strip()[:300]}

    size = dir_size_mb(gguf_out)
    ratio = fp16_mb / size
    print(f"✅ GGUF — {size:.1f} MB  ({ratio:.2f}× smaller)")
    return {"size_mb": round(size, 2), "compression": round(ratio, 2)}


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def print_summary(results: dict) -> None:
    banner("SUMMARY")
    print(f"{'Format':<10} {'Size (MB)':<14} {'Compression':<14} Status")
    print("-" * 56)
    for fmt, data in results.items():
        if "error" in data:
            print(f"{fmt:<10} {'N/A':<14} {'N/A':<14} ❌  {data['error'][:40]}")
        else:
            size = f"{data['size_mb']}"
            comp = f"{data.get('compression', 1.0):.2f}×"
            print(f"{fmt:<10} {size:<14} {comp:<14} ✅")


def write_report(results: dict, output_dir: str, base_model: str,
                 adapter_path: str | None, adapter_merged: bool) -> None:
    rows = ""
    for fmt, data in results.items():
        if "error" in data:
            rows += f"| {fmt} | N/A | N/A | ❌ Failed |\n"
        else:
            size = data["size_mb"]
            comp = f"{data.get('compression', 1.0):.2f}×"
            rows += f"| {fmt} | {size} MB | {comp} | ✅ Success |\n"

    adapter_status = "✅ Merged" if adapter_merged else ("⚠️ Architecture mismatch — not merged" if adapter_path else "None")

    report = f"""# Quantisation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `{base_model}` |
| Adapter path | `{adapter_path or 'None'}` |
| Adapter status | {adapter_status} |
| Output directory | `{output_dir}` |

## Results

| Format | Size | Compression | Status |
|--------|------|-------------|--------|
{rows}
## Output Structure

```
{output_dir}/
├── model-fp16/        # FP16 baseline (safetensors)
├── model-int8/        # BitsAndBytes INT8
├── model-int4/        # BitsAndBytes NF4 (double-quant)
└── gguf/
    └── model.gguf     # llama.cpp GGUF (f16)
```

## Quantization Methods

| Method | Bits | Key Feature |
|--------|------|-------------|
| FP16   | 16   | Full precision baseline |
| INT8   | 8    | LLM.int8() — threshold 6.0 |
| NF4    | 4    | Normalized float4 + double quant |
| GGUF   | 16   | llama.cpp portable format |

---
*Generated by DAY 3 Quantization Pipeline*
"""

    report_path = Path(output_dir) / "QUANTISATION-REPORT.md"
    report_path.write_text(report)
    print(f"✅ Report saved to {report_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    for sub in ("model-fp16", "model-int8", "model-int4", "gguf"):
        Path(args.output_dir, sub).mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    # ── FP16 ──
    fp16_path, fp16_mb, adapter_merged = save_fp16(args)
    results["FP16"] = {"size_mb": round(fp16_mb, 2)}

    tokenizer = AutoTokenizer.from_pretrained(fp16_path, trust_remote_code=True)

    # ── INT8 ──
    if not args.skip_int8:
        results["INT8"] = quantize_int8(fp16_path, tokenizer, args.output_dir, fp16_mb)

    # ── INT4 ──
    if not args.skip_int4:
        results["INT4"] = quantize_int4(fp16_path, tokenizer, args.output_dir, fp16_mb)

    # ── GGUF ──
    if not args.skip_gguf:
        results["GGUF"] = convert_gguf(fp16_path, args.output_dir, args.llama_cpp_dir, fp16_mb)

    # ── Output ──
    print_summary(results)

    results_path = Path(args.output_dir) / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"✅ Results saved to {results_path}")

    write_report(results, args.output_dir, args.base_model, args.adapter_path, adapter_merged)
    print(f"\n✅ All done. Outputs in: {args.output_dir}/\n")


if __name__ == "__main__":
    main()