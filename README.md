# Week 8 — LLM Fine-Tuning, Quantisation & Optimised Inference

> **Track:** Colab-Friendly | LoRA + QLoRA + GGUF + vLLM + llama.cpp  
> **Batch:** Hestabit 3rd Batch | Prateek

---

## 📌 Overview

This week produces engineers who can train and deploy LLMs on minimal resources. The pipeline flows end-to-end:

```
Architecture Understanding → Dataset Engineering → Fine-Tuning → Quantisation → Inference Optimisation → Production API
       (Day 1)                    (Day 1)            (Day 2)       (Day 3)           (Day 4)               (Day 5)
```

Each day's output feeds the next — fine-tuned adapters (Day 2) are quantised (Day 3), the quantised model is benchmarked (Day 4), and the optimised model is served via API (Day 5).

---

## 🎯 Week Objectives

By the end of Week 8:

- ✔ Fine-tune any LLM on Colab using LoRA / QLoRA
- ✔ Quantise models to 4× smaller size (FP16 → INT8 → INT4 → GGUF)
- ✔ Run models on laptop / CPU via llama.cpp
- ✔ Achieve 2–5× faster inference with optimisation techniques
- ✔ Deploy a production-ready local LLM API (FastAPI)
- ✔ Integrate into RAG or Agent frameworks

---

## 🤖 Approved Models (Colab-Friendly)

| Model | Size | Key Strength |
|---|---|---|
| Phi-2 / Phi-3 | 2.7B – 3.8B | Reasoning, code, math |
| Mistral 7B Instruct | 7B | General instruction quality |
| TinyLlama | 1.1B | Fast, CPU-friendly |
| Qwen | 1.5B – 4B | Multilingual, long context |

**Stack:** `transformers` · `peft` · `trl` · `accelerate` · `bitsandbytes` · `llama.cpp`

---

## 🗂️ Repository Structure

```
WEEK_8-LLM_FINE_TUNING/
├── DAY_1-LLM_ARCHITECTURE_AND_DATA_PREP_FOR_FINE_TUNING/
├── DAY_2_PEFT_LoRA_QLoRA/
├── DAY_3_QUANTISATION/
├── DAY_4_INFERENCE_OPTIMIZATION/
└── DAY_5_CAPSTONE_LOCAL_LLM_API/
```

---

## 📅 Day-by-Day Breakdown

### Day 1 — LLM Architecture + Data Prep for Fine-Tuning

**Goal:** Understand LLM internals and build a clean instruction tuning dataset.

**Key Files:**
| File | Description |
|---|---|
| `data/train.jsonl` | 1,000+ instruction samples (QA / Reasoning / Extraction) |
| `data/val.jsonl` | Held-out validation split for fine-tuning evaluation |
| `utils/data_cleaner.py` | Token length filtering, deduplication, outlier removal |
| `DATASET-ANALYSIS.md` | Token distribution analysis and dataset quality report |

**Dataset Format:**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**Dataset Composition:**
- Task Type 1 — QA pairs (question answering)
- Task Type 2 — Reasoning chains (step-by-step)
- Task Type 3 — Extraction tasks (named entity, structured output)

**Deliverables Checklist:**
- [x] 1,000+ curated JSONL samples
- [x] Token length analysis and distribution graphs
- [x] Outlier removal applied
- [x] Clean train / val split

---

### Day 2 — Parameter-Efficient Fine-Tuning (LoRA / QLoRA)

**Goal:** Fine-tune an LLM using QLoRA with only ~1% trainable parameters.

**Key Files:**
| File | Description |
|---|---|
| `notebooks/lora_train.ipynb` | QLoRA fine-tuning notebook end-to-end |
| `adapters/adapter_config.json` | LoRA rank, alpha, target module configuration |
| `adapters/tokenizer_config.json` | Tokenizer config for fine-tuned model inference |
| `training_logs/TRAINING-REPORT.md` | Loss curves, training metrics, hyperparameter report |

**QLoRA Configuration:**
```python
r         = 16       # LoRA rank
lr        = 2e-4     # learning rate
batch     = 4        # per device batch size
epochs    = 3        # training epochs
load_in_4bit = True  # 4-bit base model loading
```

**Results:**
- Trainable parameters: ~1% of total model parameters
- Memory: fits on single consumer GPU via 4-bit base loading
- Adapter weights saved separately — base model unchanged

> **Note:** Model weight binaries (`.safetensors`) are excluded from git due to size.  
> Adapter weights available on HuggingFace Hub.

**Deliverables Checklist:**
- [x] QLoRA fine-tuning notebook
- [x] Adapter config and tokenizer files
- [x] Training loss converging across epochs
- [x] Training report with hyperparameter analysis

---

### Day 3 — Quantisation (FP16 → INT8 → INT4 → GGUF)

**Goal:** Compress the fine-tuned model into progressively smaller formats and measure the tradeoffs.

**Key Files:**
| File | Description |
|---|---|
| `quantized/model-int8/` | 8-bit quantized model config and tokenizer |
| `quantized/model-int4/` | 4-bit NF4 quantized model config and tokenizer |
| `quantized/QUANTISATION-REPORT.md` | Size, speed, and quality comparison across formats |
| `scripts/quantize_model.py` | Conversion script: HuggingFace → INT8 / INT4 / GGUF |
| `scripts/evaluate_models.py` | Benchmark runner across all quantized formats |
| `benchmarks/quantization_results.json` | Raw benchmark data per format |

**Benchmark Table:**
| Format | Size | Speed | Quality |
|---|---|---|---|
| FP16 | 100% | baseline | baseline |
| INT8 | ~50% | faster | minimal loss |
| INT4 | ~25% | faster | small loss |
| GGUF q4_0 | ~25% | fastest (CPU) | small loss |

> **Note:** Model weight binaries (`.safetensors`, `.gguf`) excluded from git.  
> Quantized models available on HuggingFace Hub.

**Deliverables Checklist:**
- [x] INT8 quantized model (BitsAndBytes)
- [x] INT4 NF4 quantized model (double quant)
- [x] GGUF conversion (q4_0 / q8_0 via llama.cpp)
- [x] Quantisation report with benchmark comparison

---

### Day 4 — Inference Optimisation + Benchmarking

**Goal:** Maximize inference speed across all model formats and measure real-world performance.

**Key Files:**
| File | Description |
|---|---|
| `inference/test_inference.py` | Inference test suite across base, LoRA, INT8, INT4, GGUF |
| `benchmarks/results.csv` | Tokens/sec, VRAM, and latency per model format |
| `BENCHMARK-REPORT.md` | Full inference optimization analysis and findings |
| `scripts/benchmark_advanced.py` | KV cache, batch size, and context window benchmarks |

**Metrics Measured:**
```
✅ Tokens/sec       (throughput)
✅ VRAM / RAM usage (MB)
✅ Latency          (ms per request)
✅ Output quality   (accuracy / perplexity)
```

**Advanced Benchmarks:**
- KV Cache ON vs OFF
- Context window scaling: 32 → 128 → 512 tokens
- Batch size scaling: 1 → 2 → 4

**Optimisation Techniques Covered:**
- KV Caching — avoid recomputing attention keys/values
- llama.cpp — CPU-optimised GGUF inference
- Streaming output — token-by-token generation
- Batch inference — amortise fixed GPU overhead
- Prompt compression — shorter input = faster generation

**Deliverables Checklist:**
- [x] Inference benchmark results CSV
- [x] Test suite covering all 5 model formats
- [x] Streaming and batch inference implemented
- [x] Benchmark report with optimization analysis

---

### Day 5 — Capstone: Build & Deploy Local LLM API

**Goal:** Package the optimised quantized model into a production-ready local inference microservice.

**Key Files:**
| File | Description |
|---|---|
| `deploy/app.py` | FastAPI server with `/generate`, `/chat`, `/health`, `/models` |
| `deploy/model_loader.py` | Singleton model loader with caching on startup |
| `deploy/config.py` | Centralized server and generation parameter config |
| `deploy/schemas.py` | Pydantic request/response models for all endpoints |
| `deploy/prompt_templates.py` | Chat template formatting for instruction models |
| `Dockerfile` | Production container image with llama-cpp-python |
| `docker-compose.yml` | Model volume mount and environment configuration |
| `streamlit_ui.py` | Optional Streamlit web interface |
| `cli_chat.py` | CLI infinite chat mode |
| `docs/FINAL-REPORT.md` | Full Week 8 capstone report |

**API Endpoints:**
```
POST /generate       → Single-turn text generation
POST /chat           → Infinite multi-turn conversation
GET  /health         → Server health check
GET  /models         → List available loaded models
POST /model/switch   → Hot-swap between model formats
```

**Features:**
```
✔ Quantized model (INT4 / INT8 / GGUF)
✔ Infinite chat mode with conversation history
✔ System + user prompt templates
✔ Temperature, top-k, top-p controls
✔ JSON structured logs + unique request IDs
✔ RAG and Agent framework ready
✔ Dockerfile for containerized deployment
✔ Streamlit UI + CLI chat mode
```

**Deliverables Checklist:**
- [x] FastAPI inference server
- [x] Model loader with singleton caching
- [x] Dockerfile and docker-compose
- [x] Streamlit UI and CLI chat
- [x] Final capstone report

---

## 📊 Week 8 Completion Requirements

| Skill Area | Requirement | Status |
|---|---|---|
| Dataset | Custom + cleaned JSONL (1,000+ samples) | ✅ |
| Fine-Tuning | LoRA / QLoRA with PEFT | ✅ |
| Quantisation | INT8 + INT4 + GGUF formats | ✅ |
| Benchmarking | Speed + memory per format | ✅ |
| Inference | KV cache + streaming + batching | ✅ |
| Deployment | FastAPI server running locally | ✅ |
| Documentation | Full reports for each day | ✅ |

---

## 📖 Key Concepts Glossary

| Term | What It Is | Day |
|---|---|---|
| LoRA | Low-Rank Adaptation — trains <1% of params via rank decomposition | 2 |
| QLoRA | LoRA on a 4-bit quantised base model — fits on single GPU | 2 |
| PEFT | Umbrella library for parameter-efficient fine-tuning methods | 2 |
| BitsAndBytes | Library for INT8 / INT4 loading and dequantisation | 2, 3 |
| NF4 | NormalFloat4 — optimal 4-bit format for normally-distributed weights | 3 |
| Double Quant | Quantising the quantisation constants — saves ~0.4 bits/param | 3 |
| GGUF | GPT-Generated Unified Format — CPU-native model format for llama.cpp | 3 |
| KV Cache | Cache key-value attention pairs to avoid recomputation | 4 |
| vLLM | High-throughput LLM serving with PagedAttention | 4 |
| PagedAttention | vLLM's memory management for KV cache — reduces waste | 4 |
| Speculative Decoding | Small draft model generates candidates, large model verifies | 4 |
| SSE | Server-Sent Events — one-way token streaming from server to client | 5 |
| Singleton | Design pattern: load model once into memory, serve forever | 5 |
| apply_chat_template | Tokenizer method that formats messages into the trained prompt format | 5 |
| RAG | Retrieval-Augmented Generation — inject external context at inference | 5 |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:Prateekbit05/WEEK_8-LLM_FINE_TUNING.git
cd WEEK_8-LLM_FINE_TUNING
```

### 2. Set up virtual environment (per day)
```bash
cd DAY_X_FOLDER
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run fine-tuning (Day 2)
```bash
jupyter notebook DAY_2_PEFT_LoRA_QLoRA/notebooks/lora_train.ipynb
```

### 4. Run quantisation (Day 3)
```bash
python DAY_3_QUANTISATION/scripts/quantize_model.py
python DAY_3_QUANTISATION/scripts/evaluate_models.py
```

### 5. Run inference benchmarks (Day 4)
```bash
python DAY_4_INFERENCE_OPTIMIZATION/inference/test_inference.py
```

### 6. Start the LLM API (Day 5)
```bash
# Local
uvicorn deploy.app:app --reload --port 8000

# Docker
docker-compose up --build
```

---

