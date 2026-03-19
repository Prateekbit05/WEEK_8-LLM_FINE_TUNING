"""
============================================================
 DAY 5 CAPSTONE — app.py
 FastAPI Server — 3 Models: TinyLlama, Phi-2, Qwen2
============================================================
"""

import time
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from deploy.config import (
    config, MODEL_REGISTRY, AVAILABLE_MODELS, DEFAULT_SYSTEM_PROMPT,
)
from deploy.logger import logger, generate_request_id
from deploy.model_loader import model_manager
from deploy.schemas import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse, ChatMessageOut,
    HealthResponse, SwitchModelRequest, SwitchModelResponse,
)

SERVER_START_TIME = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SERVER_START_TIME
    logger.info("=" * 60)
    logger.info(" 🚀 LOCAL LLM API — 3 MODELS")
    logger.info(f" Models: {AVAILABLE_MODELS}")
    logger.info("=" * 60)
    try:
        model_manager.load_model(config.model.default_model)
        SERVER_START_TIME = time.time()
        logger.info("✅ Server ready!")
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise
    yield
    logger.info("🛑 Shutdown.")


app = FastAPI(
    title=config.server.api_title,
    version=config.server.api_version,
    description=config.server.api_description,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = generate_request_id()
    request.state.request_id = request_id
    start = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"[{request_id}] {response.status_code} in {duration:.1f}ms")
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "🤖 Local LLM API — TinyLlama | Phi-2 | Qwen2",
        "version": config.server.api_version,
        "current_model": model_manager.current_model_key or "none",
        "available_models": AVAILABLE_MODELS,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    if not model_manager.is_loaded():
        raise HTTPException(503, "No model loaded")
    info = model_manager.get_info()
    uptime = time.time() - SERVER_START_TIME if SERVER_START_TIME else 0
    return HealthResponse(
        status="healthy", model=info["model_name"],
        model_key=info["model_key"], model_size=info["model_size"],
        quantization=info["quantization"], device=info["device"],
        uptime_seconds=round(uptime, 2),
    )


@app.get("/info", tags=["System"])
async def info():
    if not model_manager.is_loaded():
        raise HTTPException(503, "No model loaded")
    return model_manager.get_info()


@app.get("/models", tags=["Models"])
async def list_models():
    return model_manager.list_models()


@app.post("/model/switch", response_model=SwitchModelResponse, tags=["Models"])
async def switch_model(body: SwitchModelRequest):
    model_key = body.model.value
    try:
        previous = model_manager.current_model_key or "none"
        start = time.time()
        model_manager.switch_model(model_key)
        load_time = time.time() - start
        new_info = MODEL_REGISTRY[model_key]
        return SwitchModelResponse(
            status="success", previous_model=previous,
            new_model=new_info["name"], new_model_key=model_key,
            load_time_seconds=round(load_time, 2),
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(request: Request, body: GenerateRequest):
    request_id = getattr(request.state, "request_id", generate_request_id())
    if not model_manager.is_loaded():
        raise HTTPException(503, "No model loaded")

    logger.info(f"[{request_id}] /generate [{model_manager.current_model_key}] {body.prompt[:60]}...")

    try:
        # Use 7-point system prompt if none provided
        sys_prompt = body.system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": body.prompt},
        ]

        if body.stream:
            async def stream_gen() -> AsyncGenerator[str, None]:
                full_text = ""
                count = 0
                for chunk in model_manager.stream_generate(
                    messages=messages,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k, top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ):
                    full_text += chunk
                    count += 1
                    yield json.dumps({"request_id": request_id, "model": model_manager.current_model_key, "chunk": chunk, "done": False})
                yield json.dumps({"request_id": request_id, "model": model_manager.current_model_key, "chunk": "", "done": True, "full_text": full_text.strip(), "tokens_generated": count})
            return EventSourceResponse(stream_gen())

        result = model_manager.generate(
            messages=messages,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k, top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
        )

        return GenerateResponse(
            request_id=request_id,
            model=MODEL_REGISTRY[model_manager.current_model_key]["name"],
            model_key=model_manager.current_model_key,
            prompt=body.prompt,
            generated_text=result["generated_text"],
            tokens_generated=result["tokens_generated"],
            generation_time_ms=result["generation_time_ms"],
            parameters={"temperature": body.temperature, "top_k": body.top_k, "top_p": body.top_p, "max_new_tokens": body.max_new_tokens, "repetition_penalty": body.repetition_penalty},
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(500, detail={"request_id": request_id, "error": str(e)})


@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat(request: Request, body: ChatRequest):
    request_id = getattr(request.state, "request_id", generate_request_id())
    if not model_manager.is_loaded():
        raise HTTPException(503, "No model loaded")

    try:
        messages_dict = [{"role": m.role.value, "content": m.content} for m in body.messages]

        if len(messages_dict) > config.max_chat_history:
            sys_msgs = [m for m in messages_dict if m["role"] == "system"]
            other = [m for m in messages_dict if m["role"] != "system"]
            keep = config.max_chat_history - len(sys_msgs)
            messages_dict = sys_msgs + other[-keep:]

        if body.stream:
            async def chat_stream() -> AsyncGenerator[str, None]:
                full_text = ""
                count = 0
                for chunk in model_manager.stream_generate(
                    messages=messages_dict,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k, top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                ):
                    full_text += chunk
                    count += 1
                    yield json.dumps({"request_id": request_id, "model": model_manager.current_model_key, "role": "assistant", "chunk": chunk, "done": False})
                yield json.dumps({"request_id": request_id, "model": model_manager.current_model_key, "role": "assistant", "chunk": "", "done": True, "full_text": full_text.strip(), "tokens_generated": count})
            return EventSourceResponse(chat_stream())

        result = model_manager.generate(
            messages=messages_dict,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k, top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
        )

        return ChatResponse(
            request_id=request_id,
            model=MODEL_REGISTRY[model_manager.current_model_key]["name"],
            model_key=model_manager.current_model_key,
            message=ChatMessageOut(role="assistant", content=result["generated_text"]),
            tokens_generated=result["tokens_generated"],
            generation_time_ms=result["generation_time_ms"],
            total_messages=len(body.messages) + 1,
            parameters={"temperature": body.temperature, "top_k": body.top_k, "top_p": body.top_p, "max_new_tokens": body.max_new_tokens, "repetition_penalty": body.repetition_penalty},
        )
    except Exception as e:
        logger.error(f"[{request_id}] Chat error: {e}")
        raise HTTPException(500, detail={"request_id": request_id, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" 🚀 LOCAL LLM API")
    print(f" Models: {AVAILABLE_MODELS}")
    print(f" Port:   {config.server.port}")
    print(f" Docs:   http://localhost:{config.server.port}/docs")
    print("=" * 60)
    uvicorn.run("deploy.app:app", host=config.server.host, port=config.server.port, workers=config.server.workers, log_level=config.server.log_level, reload=False)