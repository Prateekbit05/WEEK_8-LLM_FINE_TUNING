"""
============================================================
 DAY 5 CAPSTONE — config.py
 3-Model Configuration: TinyLlama, Phi-2, Qwen2
============================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
#  MODEL REGISTRY — 3 MODELS ONLY
# ============================================================
MODEL_REGISTRY: Dict[str, dict] = {
    "tinyllama": {
        "name": "TinyLlama 1.1B Chat",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "tinyllama",
        "size": "1.1B",
        "parameters": "1.1 Billion",
        "ram_required_gb": 5,
        "description": "Fastest model. Best for CPU. Chat-optimized LLaMA architecture.",
        "default_max_tokens": 512,
        "context_length": 2048,
        "strengths": "Speed, low memory, chat format",
        "weaknesses": "Lower quality, can hallucinate",
    },
    "phi2": {
        "name": "Microsoft Phi-2",
        "model_id": "microsoft/phi-2",
        "family": "phi",
        "size": "2.7B",
        "parameters": "2.7 Billion",
        "ram_required_gb": 12,
        "description": "Best quality/size ratio. Trained on textbook-quality data by Microsoft.",
        "default_max_tokens": 512,
        "context_length": 2048,
        "strengths": "Reasoning, code, math, quality",
        "weaknesses": "No native chat template, needs 12GB RAM",
    },
    "qwen": {
        "name": "Qwen2 1.5B Instruct",
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
        "family": "qwen",
        "size": "1.5B",
        "parameters": "1.5 Billion",
        "ram_required_gb": 7,
        "description": "Multilingual model by Alibaba. Good for English + Chinese + diverse tasks.",
        "default_max_tokens": 512,
        "context_length": 32768,
        "strengths": "Multilingual, long context, instruction following",
        "weaknesses": "Slightly lower English quality than Phi-2",
    },
}

AVAILABLE_MODELS: List[str] = list(MODEL_REGISTRY.keys())


# ============================================================
#  DEFAULT SYSTEM PROMPT — 7-POINT FORMAT
# ============================================================
DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable and detailed AI assistant. "
    "Always structure your responses in exactly 7 numbered points. "
    "Each point should be a complete sentence with detailed explanation. "
    "Format your response as:\n"
    "1. [First point with detailed explanation]\n"
    "2. [Second point with detailed explanation]\n"
    "3. [Third point with detailed explanation]\n"
    "4. [Fourth point with detailed explanation]\n"
    "5. [Fifth point with detailed explanation]\n"
    "6. [Sixth point with detailed explanation]\n"
    "7. [Seventh point with detailed explanation]\n\n"
    "Make each point informative, detailed, and educational. "
    "If the user greets you, greet them warmly and share "
    "7 interesting points about what you can help with. "
    "Never give short answers. Always provide 7 detailed numbered points."
)


@dataclass
class ModelConfig:
    """Model loading configuration"""
    default_model: str = "tinyllama"
    quantization: str = "none"
    max_model_len: int = 2048
    device_map: str = "auto"
    trust_remote_code: bool = True
    cache_dir: str = os.path.expanduser("~/.cache/huggingface/hub")


@dataclass
class GenerationConfig:
    """Default generation parameters — for long detailed responses"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_title: str = "Multi-Model Local LLM API"
    api_version: str = "2.0.0"
    api_description: str = (
        "Local LLM API supporting TinyLlama 1.1B, "
        "Phi-2 2.7B, and Qwen2 1.5B with 7-point responses."
    )


@dataclass
class AppConfig:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log_dir: str = "logs"
    max_chat_history: int = 20


config = AppConfig()


def get_model_info(model_key: str) -> dict:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_key}. Available: {AVAILABLE_MODELS}"
        )
    return MODEL_REGISTRY[model_key]


def get_model_id(model_key: str) -> str:
    return get_model_info(model_key)["model_id"]


if __name__ == "__main__":
    print("=" * 60)
    print(" 3-MODEL CONFIGURATION")
    print("=" * 60)
    print(f"\n  Default Model:      {config.model.default_model}")
    print(f"  Server Port:        {config.server.port}")
    print(f"  Max New Tokens:     {config.generation.max_new_tokens}")
    print(f"  Temperature:        {config.generation.temperature}")
    print(f"  Repetition Penalty: {config.generation.repetition_penalty}")
    print(f"\n  Available Models ({len(AVAILABLE_MODELS)}):")
    print(f"  {'='*50}")
    for key, info in MODEL_REGISTRY.items():
        print(f"\n  [{key}]")
        print(f"    Name:       {info['name']}")
        print(f"    Model ID:   {info['model_id']}")
        print(f"    Size:       {info['size']} ({info['parameters']})")
        print(f"    RAM:        ~{info['ram_required_gb']}GB")
        print(f"    Context:    {info['context_length']} tokens")
        print(f"    Strengths:  {info['strengths']}")
        print(f"    Weaknesses: {info['weaknesses']}")
        print(f"    Desc:       {info['description']}")
    print(f"\n{'='*60}")