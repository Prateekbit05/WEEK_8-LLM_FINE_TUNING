"""
============================================================
 DAY 5 CAPSTONE — prompt_templates.py
 Uses tokenizer.apply_chat_template() — correct for all models
============================================================
"""

from typing import List, Dict


def build_generate_messages(
    user_prompt: str,
    system_prompt: str = None,
) -> List[Dict[str, str]]:
    """Build messages list for /generate endpoint"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_chat_messages(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Pass through for /chat endpoint"""
    return messages


if __name__ == "__main__":
    msgs = build_generate_messages("What is Python?", "Be helpful.")
    for m in msgs:
        print(f"  {m['role']}: {m['content']}")