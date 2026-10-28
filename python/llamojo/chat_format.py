""" Chat formatters and template engines for llamojo matching llama_cpp.llama_chat_format."""

import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import gguf

try:
    import jinja2
    _JINJA2_AVAILABLE = True
except ImportError:
    jinja2 = None  # type: ignore
    _JINJA2_AVAILABLE = False


@dataclasses.dataclass
class ChatFormatterResponse:
    """ Dataclass storing formatted prompt and stopping parameters."""
    prompt: str
    stop: Optional[Union[str, List[str]]] = None
    stopping_criteria: Optional[Any] = None
    added_special: bool = False


# Type definition for a chat formatter callable
ChatFormatter = Callable[..., ChatFormatterResponse]

_CHAT_FORMATTERS: Dict[str, ChatFormatter] = {}


def register_chat_format(name: str):
    """ Decorator to register a chat formatter function by name."""
    def decorator(fn: ChatFormatter) -> ChatFormatter:
        _CHAT_FORMATTERS[name] = fn
        return fn
    return decorator


def get_chat_formatter(name: str) -> Optional[ChatFormatter]:
    """ Retrieve a registered chat formatter by name."""
    return _CHAT_FORMATTERS.get(name)


def _get_system_message(messages: Sequence[Dict[str, Any]]) -> str:
    """ Extract system message content from messages list."""
    for m in messages:
        if m.get("role") == "system":
            return str(m.get("content") or "")
    return ""


# ==============================================================================
# Built-in Formatters
# ==============================================================================

@register_chat_format("chatml")
def format_chatml(messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ChatFormatterResponse:
    """ Standard ChatML formatting (<|im_start|>role\ncontent<|im_end|>)."""
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return ChatFormatterResponse(prompt=prompt, stop=["<|im_end|>"])


@register_chat_format("llama-2")
def format_llama2(messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ChatFormatterResponse:
    """ Llama 2 [INST] prompt format."""
    system_msg = _get_system_message(messages)
    prompt = ""
    is_first_user = True
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            continue
        elif role == "user":
            if is_first_user and system_msg:
                prompt += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{content} [/INST]"
                is_first_user = False
            else:
                prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} </s>"
    return ChatFormatterResponse(prompt=prompt, stop=["</s>"])


@register_chat_format("llama-3")
def format_llama3(messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ChatFormatterResponse:
    """ Llama 3 <|start_header_id|> format."""
    prompt = "<|begin_of_text|>"
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return ChatFormatterResponse(prompt=prompt, stop=["<|eot_id|>"])


@register_chat_format("gemma")
def format_gemma(messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ChatFormatterResponse:
    """ Gemma / Gemma 2 <start_of_turn> format."""
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            # Gemma ignores system prompt or folds it into first user turn
            prompt += f"<start_of_turn>user\n{content}\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        else:
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return ChatFormatterResponse(prompt=prompt, stop=["<end_of_turn>"])


@register_chat_format("mistral-instruct")
def format_mistral_instruct(messages: Sequence[Dict[str, Any]], **kwargs: Any) -> ChatFormatterResponse:
    """ Mistral Instruct [INST] format."""
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            prompt += f"[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content}</s>"
    return ChatFormatterResponse(prompt=prompt, stop=["</s>"])


# ==============================================================================
# Jinja2 Chat Formatter Engine
# ==============================================================================

class Jinja2ChatFormatter:
    """ Dynamic Jinja2 template renderer matching Hugging Face / GGUF tokenizer.chat_template."""

    def __init__( self,
            template: str,
            eos_token: str = "<|im_end|>",
            bos_token: str = "<|im_start|>",
            add_generation_prompt: bool = True,
            stop_tokens: Optional[List[str]] = None,
        ):
        if not _JINJA2_AVAILABLE:
            raise ImportError(
                "jinja2 is required for Jinja2ChatFormatter. "
                "Install it with: pip install jinja2"
            )

        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.add_generation_prompt = add_generation_prompt
        self.stop_tokens = stop_tokens or [eos_token]

        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["tojson"] = lambda x: json.dumps(x)
        self._compiled = env.from_string(template)

    def __call__( self,
            messages: Sequence[Dict[str, Any]],
            **kwargs: Any,
        ) -> ChatFormatterResponse:
        def raise_exception(message: str):
            raise ValueError(message)

        prompt = self._compiled.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=self.add_generation_prompt,
            **kwargs,
        )
        return ChatFormatterResponse(
            prompt=prompt,
            stop=self.stop_tokens,
            added_special=True,
        )


def extract_chat_template_from_gguf(gguf_path: str) -> Optional[str]:
    """ Extract `tokenizer.chat_template` string directly from a GGUF file if available."""
    try:
        reader = gguf.GGUFReader(gguf_path)
        field = reader.fields.get("tokenizer.chat_template")
        if field and field.parts:
            val = field.parts[-1]
            return bytes(val).decode("utf-8", errors="replace")
    except Exception:
        pass
    return None
