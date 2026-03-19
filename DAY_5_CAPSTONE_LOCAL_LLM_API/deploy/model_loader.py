"""
============================================================
 DAY 5 CAPSTONE — model_loader.py
 Multi-Model Manager — 3 Models
 
 Models:
   1. TinyLlama 1.1B Chat
   2. Microsoft Phi-2 2.7B
   3. Qwen2 1.5B Instruct
   
 Features:
   - apply_chat_template() for correct formatting
   - Hot-swap between models
   - Singleton pattern (load once)
   - Long detailed response support
============================================================
"""

import gc
import torch
import time
import threading
from typing import Generator, List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from deploy.config import (
    config,
    MODEL_REGISTRY,
    AVAILABLE_MODELS,
    get_model_info,
    DEFAULT_SYSTEM_PROMPT,
)
from deploy.logger import logger


# ============================================================
#  OUTPUT CLEANING
# ============================================================
CUT_MARKERS = [
    "\nUser:", "\nHuman:", "\nSystem:",
    "\n\nUser:", "\n\nHuman:", "\n\nSystem:",
    "\n<|user|>", "\n<|system|>",
    "\n<|im_start|>user", "\n<|im_start|>system",
    "\n### User:", "\n### System:",
]

ARTIFACTS = [
    "<|im_end|>", "<|im_start|>", "</s>", "<s>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "<|endoftext|>", "<|im_start|>assistant",
    "### Assistant:", "### User:", "### System:",
]


def clean_generated_text(text: str) -> str:
    """Remove artifacts and cut at fake role markers"""
    if not text:
        return ""

    for artifact in ARTIFACTS:
        text = text.replace(artifact, "")

    for marker in CUT_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    text_lower = text.lower()
    for marker in ["\nuser:", "\nhuman:", "\nsystem:"]:
        idx = text_lower.find(marker)
        if idx > 0:
            text = text[:idx]

    return text.strip()


# ============================================================
#  MULTI-MODEL MANAGER (SINGLETON)
# ============================================================
class MultiModelManager:
    """
    Manages 3 LLM models: TinyLlama, Phi-2, Qwen2.
    Loads ONE at a time. Supports hot-swapping.
    Uses apply_chat_template() for correct prompt formatting.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.device = None
        self.current_model_key = None
        self.current_model_id = None
        self.current_family = None
        self.load_time = None
        self._has_chat_template = False
        self._initialized = True

    # --------------------------------------------------------
    #  FORMAT PROMPT USING TOKENIZER
    # --------------------------------------------------------
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages using tokenizer's apply_chat_template().
        This is the CORRECT way — each model knows its own format.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded!")

        if self._has_chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}")

        # Fallback for Phi-2 (no chat template)
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"Instruct: {content}\n\n"
            elif role == "user":
                formatted += f"{content}\nOutput:"
            elif role == "assistant":
                formatted += f" {content}\n\n"

        return formatted

    # --------------------------------------------------------
    #  LOAD MODEL
    # --------------------------------------------------------
    def load_model(self, model_key: str = None) -> dict:
        """Load one of 3 models: tinyllama, phi2, qwen"""

        if model_key is None:
            model_key = config.model.default_model

        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: '{model_key}'. "
                f"Available: {AVAILABLE_MODELS}"
            )

        model_info = get_model_info(model_key)
        model_id = model_info["model_id"]

        # Already loaded?
        if self.current_model_key == model_key and self.model is not None:
            logger.info(f"'{model_key}' already loaded.")
            return {"status": "already_loaded", "model_key": model_key}

        logger.info("=" * 55)
        logger.info(f"  LOADING: {model_info['name']}")
        logger.info(f"  Size:    {model_info['size']} ({model_info['parameters']})")
        logger.info(f"  ID:      {model_id}")
        logger.info(f"  RAM:     ~{model_info['ram_required_gb']}GB")
        logger.info("=" * 55)

        start_time = time.time()

        # Unload previous
        self._unload_current()

        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            self.device = "cpu"
            logger.info("Device: CPU")

        # Quantization
        quantization_config = None
        if config.model.quantization == "4bit" and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Quantization: 4-bit NF4")
            except ImportError:
                logger.warning("bitsandbytes not available")
        else:
            logger.info("Quantization: None (full precision)")

        # ---- TOKENIZER ----
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Check chat template
        self._has_chat_template = False
        try:
            test_msg = [{"role": "user", "content": "test"}]
            self.tokenizer.apply_chat_template(
                test_msg, tokenize=False, add_generation_prompt=True
            )
            self._has_chat_template = True
            logger.info(f"Chat template: ✅ YES")
        except Exception:
            logger.info(f"Chat template: ❌ NO (using fallback)")

        # ---- MODEL ----
        logger.info("Loading model weights...")

        load_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": config.model.trust_remote_code,
            "cache_dir": config.model.cache_dir,
        }

        if quantization_config and self.device == "cuda":
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

        # State
        self.current_model_key = model_key
        self.current_model_id = model_id
        self.current_family = model_info["family"]
        self.load_time = time.time() - start_time

        param_count = sum(p.numel() for p in self.model.parameters())

        logger.info(f"✅ {model_info['name']} loaded!")
        logger.info(f"   Parameters:    {param_count / 1e6:.1f}M")
        logger.info(f"   Load time:     {self.load_time:.2f}s")
        logger.info(f"   Device:        {self.device}")
        logger.info(f"   Chat template: {self._has_chat_template}")

        return {
            "status": "loaded",
            "model_key": model_key,
            "model_name": model_info["name"],
            "model_id": model_id,
            "family": self.current_family,
            "parameters": param_count,
            "load_time_seconds": round(self.load_time, 2),
            "device": self.device,
            "has_chat_template": self._has_chat_template,
        }

    # --------------------------------------------------------
    #  UNLOAD
    # --------------------------------------------------------
    def _unload_current(self):
        if self.model is not None:
            logger.info(f"Unloading: {self.current_model_key}")
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.current_model_key = None
        self.current_model_id = None
        self.current_family = None
        self._has_chat_template = False
        logger.info("Model unloaded, memory freed.")

    # --------------------------------------------------------
    #  SWITCH
    # --------------------------------------------------------
    def switch_model(self, model_key: str) -> dict:
        previous = self.current_model_key
        logger.info(f"Switching: {previous} → {model_key}")
        result = self.load_model(model_key)
        result["previous_model"] = previous
        return result

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # --------------------------------------------------------
    #  GENERATE
    # --------------------------------------------------------
    def generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
    ) -> dict:
        """Generate text from messages or prompt"""

        if not self.is_loaded():
            raise RuntimeError("No model loaded!")

        max_new_tokens = max_new_tokens or config.generation.max_new_tokens
        temperature = temperature if temperature is not None else config.generation.temperature
        top_k = top_k or config.generation.top_k
        top_p = top_p if top_p is not None else config.generation.top_p
        repetition_penalty = repetition_penalty or config.generation.repetition_penalty

        start_time = time.time()

        # Build prompt
        if messages is not None:
            formatted_prompt = self.format_prompt(messages)
        elif prompt is not None:
            formatted_prompt = prompt
        else:
            raise ValueError("Provide 'prompt' or 'messages'")

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.model.max_model_len - max_new_tokens,
        )

        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_length = input_ids.shape[1]

        # Stop tokens (only EOS — allow long generation)
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        for s in ["</s>", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids and ids[0] not in eos_ids:
                    eos_ids.append(ids[0])
            except Exception:
                pass
        if not eos_ids:
            eos_ids = [2]

        # Generation kwargs
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_ids,
            "repetition_penalty": repetition_penalty,
        }

        effective_temp = max(temperature, 0.01)
        if effective_temp > 0.01:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = effective_temp
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        new_tokens = outputs[0][input_length:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_text = clean_generated_text(raw_text)

        tokens_generated = len(new_tokens)
        generation_time = (time.time() - start_time) * 1000

        tps = round(tokens_generated / (generation_time / 1000), 2) if generation_time > 0 else 0

        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time_ms": round(generation_time, 2),
            "tokens_per_second": tps,
        }

    # --------------------------------------------------------
    #  STREAM GENERATE
    # --------------------------------------------------------
    def stream_generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
    ) -> Generator[str, None, None]:
        """Stream tokens one by one"""

        if not self.is_loaded():
            raise RuntimeError("No model loaded!")

        max_new_tokens = max_new_tokens or config.generation.max_new_tokens
        temperature = temperature if temperature is not None else config.generation.temperature
        top_k = top_k or config.generation.top_k
        top_p = top_p if top_p is not None else config.generation.top_p
        repetition_penalty = repetition_penalty or config.generation.repetition_penalty

        if messages is not None:
            formatted_prompt = self.format_prompt(messages)
        elif prompt is not None:
            formatted_prompt = prompt
        else:
            raise ValueError("Provide 'prompt' or 'messages'")

        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True,
            max_length=config.model.max_model_len - max_new_tokens,
        )

        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        for s in ["</s>", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids and ids[0] not in eos_ids:
                    eos_ids.append(ids[0])
            except Exception:
                pass
        if not eos_ids:
            eos_ids = [2]

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_ids,
            "streamer": streamer,
            "repetition_penalty": repetition_penalty,
        }

        effective_temp = max(temperature, 0.01)
        if effective_temp > 0.01:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = effective_temp
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        thread = threading.Thread(target=self._gen_thread, args=(gen_kwargs,), daemon=True)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            should_stop = any(m in accumulated for m in CUT_MARKERS)
            if should_stop:
                break
            clean_chunk = chunk
            for art in ARTIFACTS:
                clean_chunk = clean_chunk.replace(art, "")
            if clean_chunk:
                yield clean_chunk

        thread.join(timeout=180)

    def _gen_thread(self, gen_kwargs):
        try:
            with torch.no_grad():
                self.model.generate(**gen_kwargs)
        except Exception as e:
            logger.error(f"Gen thread error: {e}")

    # --------------------------------------------------------
    #  INFO & LIST
    # --------------------------------------------------------
    def get_info(self) -> dict:
        if not self.is_loaded():
            return {"status": "no model loaded"}
        reg = MODEL_REGISTRY.get(self.current_model_key, {})
        try:
            device = str(next(self.model.parameters()).device)
        except StopIteration:
            device = self.device
        return {
            "model_key": self.current_model_key,
            "model_name": reg.get("name", self.current_model_id),
            "model_id": self.current_model_id,
            "model_size": reg.get("size", "?"),
            "model_family": reg.get("family", "?"),
            "quantization": config.model.quantization,
            "device": device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "load_time_seconds": round(self.load_time, 2) if self.load_time else 0,
            "vocab_size": self.tokenizer.vocab_size,
            "has_chat_template": self._has_chat_template,
            "context_length": reg.get("context_length", "?"),
            "description": reg.get("description", ""),
        }

    def list_models(self) -> dict:
        return {
            "current_model_key": self.current_model_key,
            "current_model": (
                MODEL_REGISTRY.get(self.current_model_key, {}).get("name", "None")
                if self.current_model_key else "None"
            ),
            "total_models": len(MODEL_REGISTRY),
            "available_models": {
                key: {
                    "name": info["name"],
                    "size": info["size"],
                    "family": info["family"],
                    "ram_required_gb": info["ram_required_gb"],
                    "description": info["description"],
                    "loaded": key == self.current_model_key,
                }
                for key, info in MODEL_REGISTRY.items()
            }
        }


# Global instance
model_manager = MultiModelManager()


# ============================================================
#  TEST ALL 3 MODELS
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" 3-MODEL LOADER TEST")
    print(" Models: TinyLlama 1.1B | Phi-2 2.7B | Qwen2 1.5B")
    print("=" * 60)

    # List models
    models = model_manager.list_models()
    print(f"\nAvailable Models ({models['total_models']}):")
    for key, info in models["available_models"].items():
        print(f"  [{key}] {info['name']} ({info['size']}) — {info['description'][:50]}...")

    # Load default (TinyLlama)
    print("\n" + "=" * 60)
    print(" Loading TinyLlama 1.1B...")
    print("=" * 60)
    model_manager.load_model("tinyllama")

    info = model_manager.get_info()
    print(f"  Name:          {info['model_name']}")
    print(f"  Chat template: {info['has_chat_template']}")
    print(f"  Parameters:    {info['parameters']/1e6:.0f}M")

    # Test generation
    print("\n" + "=" * 60)
    print(" TEST: 'What is Python?' (7-point format)")
    print("=" * 60)

    test_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "What is Python?"},
    ]

    result = model_manager.generate(messages=test_messages, max_new_tokens=400)
    print(f"\n{result['generated_text']}")
    print(f"\n  Tokens: {result['tokens_generated']}")
    print(f"  Time:   {result['generation_time_ms']:.0f}ms")
    print(f"  Speed:  {result['tokens_per_second']} tok/s")

    print("\n✅ Test complete!")