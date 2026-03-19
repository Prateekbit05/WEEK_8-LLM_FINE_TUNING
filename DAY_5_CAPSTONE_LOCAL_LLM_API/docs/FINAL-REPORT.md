# FINAL CAPSTONE REPORT — Week 8 Day 5

**Student:** Prateek
**Organization:** Hestabit (3rd Batch)
**Project:** Multi-Model Local LLM Inference API
**Models:** TinyLlama 1.1B | Phi-2 2.7B | Qwen2 1.5B
**Date:** 2026-03-05

---

## 1. Project Summary

Built a production-ready Local LLM API serving **3 different models** (TinyLlama 1.1B, Phi-2 2.7B, Qwen2 1.5B) via FastAPI. The system integrates all Week 8 concepts — transformer architecture, tokenization, fine-tuning context, inference optimization, and production deployment — into a single deployable microservice.

**Key capabilities:**
- Hot-swapping between models at runtime without server restart
- Single-prompt (`/generate`) and multi-turn infinite chat (`/chat`)
- SSE streaming for real-time token delivery
- 7-point structured response format
- JSON structured logging with request ID tracking
- Docker containerization
- Ready for RAG and Agent framework integration

---

## 2. Models Used

### Model 1: TinyLlama 1.1B Chat

| Property | Value |
|----------|-------|
| Registry Key | `tinyllama` |
| HuggingFace ID | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 Billion |
| Architecture | LLaMA decoder-only |
| RAM (CPU, FP32) | ~5 GB |
| VRAM (GPU, FP16) | ~2.2 GB |
| VRAM (GPU, 4-bit) | ~0.8 GB |
| Context Length | 2,048 tokens |
| Speed (CPU) | ~4–6 tok/s |
| Chat Template | ✅ `<|user|>` format |
| License | Apache 2.0 |

### Model 2: Microsoft Phi-2

| Property | Value |
|----------|-------|
| Registry Key | `phi2` |
| HuggingFace ID | `microsoft/phi-2` |
| Parameters | 2.7 Billion |
| Architecture | Transformer decoder-only |
| RAM (CPU, FP32) | ~12 GB |
| VRAM (GPU, FP16) | ~5.4 GB |
| VRAM (GPU, 4-bit) | ~1.8 GB |
| Context Length | 2,048 tokens |
| Speed (CPU) | ~2–4 tok/s |
| Chat Template | ❌ `Instruct:/Output:` fallback |
| License | MIT |

### Model 3: Qwen2 1.5B Instruct

| Property | Value |
|----------|-------|
| Registry Key | `qwen` |
| HuggingFace ID | `Qwen/Qwen2-1.5B-Instruct` |
| Parameters | 1.5 Billion |
| Architecture | Transformer with RoPE + GQA |
| RAM (CPU, FP32) | ~7 GB |
| VRAM (GPU, FP16) | ~3 GB |
| VRAM (GPU, 4-bit) | ~1.2 GB |
| Context Length | 32,768 tokens |
| Speed (CPU) | ~3–5 tok/s |
| Chat Template | ✅ `<|im_start|>` format |
| License | Apache 2.0 |

---

## 3. Architecture Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Framework | FastAPI | Async, auto-docs, Pydantic validation |
| Models | 3 (1.1B–2.7B) | CPU-friendly parameter range |
| Model management | Singleton + hot-swap | Memory efficient; one model at a time |
| Prompting | `apply_chat_template()` | Correct format per model family |
| Quantization | 4-bit NF4 (GPU only) | ~75% VRAM savings |
| Streaming | SSE (Server-Sent Events) | Real-time token delivery |
| Logging | JSON structured | Production-grade, machine-parseable |
| Output | Stop sequences + post-processing | Clean, truncated responses |

---

## 4. Features Implemented

### Required Features
- [x] `POST /generate` — Single prompt generation
- [x] `POST /chat` — Multi-turn infinite chat
- [x] Quantized model support (4-bit/8-bit on GPU)
- [x] Infinite chat mode (append messages without limit)
- [x] System + user prompts
- [x] Temperature, top-k, top-p, repetition penalty controls
- [x] JSON structured logs with unique request IDs
- [x] RAG / Agent ready API design

### Multi-Model Features
- [x] 3 models: TinyLlama, Phi-2, Qwen2
- [x] `GET /models` — list all models with status
- [x] `POST /model/switch` — hot-swap at runtime
- [x] `tokenizer.apply_chat_template()` per model
- [x] 7-point structured response format

### Bonus Features
- [x] Dockerfile + docker-compose
- [x] CLI chat (`cli_chat.py`) with slash commands
- [x] Streamlit web UI (`streamlit_ui.py`)
- [x] SSE streaming
- [x] Singleton model caching
- [x] CORS middleware
- [x] Swagger auto-docs at `/docs`
- [x] X-Request-ID in response headers
- [x] Output artifact removal and stop marker truncation
- [x] Comprehensive test suite (10 tests)

---

## 5. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome + endpoint list |
| `GET` | `/health` | Health check with model info |
| `GET` | `/info` | Detailed model metadata |
| `GET` | `/models` | List all 3 models |
| `POST` | `/model/switch` | Switch active model |
| `POST` | `/generate` | Single prompt → text |
| `POST` | `/chat` | Multi-turn conversation |

---

## 6. Performance (CPU — No GPU)

| Metric | TinyLlama 1.1B | Phi-2 2.7B | Qwen2 1.5B |
|--------|----------------|------------|------------|
| Model load time | ~3 s | ~8–12 s | ~5–7 s |
| RAM usage | ~5 GB | ~12 GB | ~7 GB |
| Tokens/sec | ~4–6 | ~2–4 | ~3–5 |
| 100-token response | ~20–25 s | ~30–50 s | ~25–35 s |
| Model switch time | ~5 s | ~10–15 s | ~7–10 s |

---

## 7. Key Technical Problems & Solutions

### Problem 1: Garbage / Off-format Output
**Cause:** Manual prompt templates didn't match the model's training format.
**Fix:** Used `tokenizer.apply_chat_template()` — each tokenizer knows exactly which format it was trained on.
**Result:** Clean, contextual, relevant responses.

### Problem 2: Model Rambling / Fake Dialogue Turns
**Cause:** No stop sequences set; model generates `User:` / `Human:` continuations.
**Fix:** Pass EOS token IDs in `eos_token_id` and post-process to truncate at role markers.
**Result:** Responses stop naturally at the correct point.

### Problem 3: Memory OOM on Model Switch
**Cause:** Old model not freed before loading the new one.
**Fix:** `del model` → `gc.collect()` → `torch.cuda.empty_cache()` before loading.
**Result:** Clean model swaps without OOM errors.

### Problem 4: Port Conflict on Startup
**Cause:** Another process already bound to port 8000.
**Fix:** Changed server port to 8001.
**Result:** No conflicts.

### Problem 5: Variable Scope Error in CLI
**Cause:** List comprehension referencing a variable before it was created in the same scope.
**Fix:** Rewrote using a conventional loop.
**Result:** CLI works correctly.

---

## 8. Production Readiness

| Feature | Status |
|---------|--------|
| Structured JSON logging | ✅ |
| Request ID tracking | ✅ |
| HTTP error codes (200/400/422/500/503) | ✅ |
| Input validation (Pydantic) | ✅ |
| CORS support | ✅ |
| Health check endpoint | ✅ |
| Docker containerization | ✅ |
| Configuration management | ✅ |
| Singleton model caching | ✅ |
| Graceful startup/shutdown (lifespan) | ✅ |
| Swagger auto-docs | ✅ |

---

## 9. RAG / Agent Integration Examples

### RAG Integration

```python
import requests

def query_with_context(context: str, question: str) -> str:
    resp = requests.post("http://localhost:8001/generate", json={
        "prompt": f"Context:\n{context}\n\nQuestion: {question}",
        "system_prompt": "Answer only based on the provided context.",
        "max_new_tokens": 300,
    })
    return resp.json()["generated_text"]

answer = query_with_context(
    context="Python was created by Guido van Rossum in 1991.",
    question="Who created Python?"
)
```

### Agent Integration

```python
def agent_step(history: list, tool_result: str = None) -> str:
    if tool_result:
        history.append({"role": "user", "content": f"Tool result: {tool_result}"})
    resp = requests.post("http://localhost:8001/chat", json={
        "messages": history,
        "max_new_tokens": 300,
    })
    return resp.json()["message"]["content"]
```

### Model Selection Per Task

```python
# Fast prototyping → TinyLlama
requests.post("http://localhost:8001/model/switch", json={"model": "tinyllama"})

# Reasoning / code → Phi-2
requests.post("http://localhost:8001/model/switch", json={"model": "phi2"})

# Multilingual / long context → Qwen2
requests.post("http://localhost:8001/model/switch", json={"model": "qwen"})
```

---

## 10. Week 8 Learning Summary

| Day | Topic | Key Takeaway |
|-----|-------|-------------|
| 1 | Transformer Architecture | Self-attention, multi-head attention, positional encoding |
| 2 | Tokenization & Prompting | BPE, prompt engineering, system prompts |
| 3 | Fine-Tuning (LoRA/QLoRA) | Parameter-efficient adaptation with PEFT |
| 4 | Inference Optimization | Quantization, KV cache (17.79×), batching |
| 5 | Capstone: 3-Model API | Production deployment integrating all concepts |

---

## 11. Files Delivered

| File | Description |
|------|-------------|
| `deploy/app.py` | FastAPI server |
| `deploy/model_loader.py` | 3-model singleton manager |
| `deploy/config.py` | Config + model registry |
| `deploy/schemas.py` | Pydantic request/response models |
| `deploy/prompt_templates.py` | Prompt formatting per model |
| `deploy/logger.py` | JSON structured logging |
| `cli_chat.py` | Rich CLI chat client |
| `streamlit_ui.py` | Streamlit web UI |
| `test_api.py` | Test suite (10 tests) |
| `Dockerfile` | Container image build |
| `docker-compose.yml` | Docker compose config |
| `README.md` | Project overview |
| `FINAL-REPORT.md` | This report |
| `COMMANDS.md` | All shell commands |
| `CURL-COMMANDS.md` | curl reference |
| `POSTMAN-COMMANDS.md` | Postman guide |
| `MODEL-INFO.md` | Detailed model specs |
| `TOPICS-INFO.md` | Topics and concepts |
| `DOCKER.md` | Docker guide |

---

## 12. Conclusion

Successfully built and deployed a 3-model local LLM inference microservice that:

- Serves TinyLlama 1.1B, Phi-2 2.7B, and Qwen2 1.5B via a unified API
- Supports hot-swapping between models at runtime
- Provides clean structured responses using `apply_chat_template()`
- Handles single-turn and infinite multi-turn chat
- Streams tokens in real-time via SSE
- Is production-ready with logging, validation, error handling, and Docker
- Is architecturally ready for RAG and Agent frameworks

This capstone integrates all Week 8 concepts into a single deployable system.

---

