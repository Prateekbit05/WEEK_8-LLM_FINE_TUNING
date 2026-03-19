"""
============================================================
 DAY 5 CAPSTONE — schemas.py
 Pydantic models — 3 Models: TinyLlama, Phi-2, Qwen2
============================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ModelName(str, Enum):
    """Only 3 models available"""
    tinyllama = "tinyllama"
    phi2 = "phi2"
    qwen = "qwen"


class GenerateRequest(BaseModel):
    """POST /generate"""
    prompt: str = Field(..., description="Input prompt")
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt (uses 7-point format by default)"
    )
    max_new_tokens: Optional[int] = Field(default=512, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=200)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.15, ge=1.0, le=3.0)
    stream: Optional[bool] = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is machine learning?",
                "max_new_tokens": 512,
                "temperature": 0.7,
            }
        }


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    """POST /chat"""
    messages: List[ChatMessage] = Field(...)
    max_new_tokens: Optional[int] = Field(default=512, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=200)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.15, ge=1.0, le=3.0)
    stream: Optional[bool] = Field(default=False)


class SwitchModelRequest(BaseModel):
    """POST /model/switch"""
    model: ModelName = Field(..., description="Model: tinyllama, phi2, or qwen")

    class Config:
        json_schema_extra = {
            "example": {"model": "phi2"}
        }


class ChatMessageOut(BaseModel):
    role: str
    content: str


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    model_key: str
    prompt: str
    generated_text: str
    tokens_generated: int
    generation_time_ms: float
    parameters: Dict[str, Any]


class ChatResponse(BaseModel):
    request_id: str
    model: str
    model_key: str
    message: ChatMessageOut
    tokens_generated: int
    generation_time_ms: float
    total_messages: int
    parameters: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_key: str
    model_size: str
    quantization: str
    device: str
    uptime_seconds: float


class SwitchModelResponse(BaseModel):
    status: str
    previous_model: str
    new_model: str
    new_model_key: str
    load_time_seconds: float