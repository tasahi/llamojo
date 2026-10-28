# Model Chat Template Engines in `llamojo`

## Overview
`llamojo` implements a dedicated chat formatting engine matching `llama_cpp.llama_chat_format`. It enables both **automatic extraction of Jinja2 templates directly from GGUF metadata** and **explicit selection of popular architectural chat templates** (`chatml`, `llama-2`, `llama-3`, `gemma`, `mistral-instruct`).

---

## 1. Automatic GGUF Chat Template Extraction
When initializing a `Llama` instance, `llamojo` automatically inspects the `.gguf` file header for `tokenizer.chat_template`. If found, a dynamic `Jinja2ChatFormatter` is constructed using the model's exact BOS and EOS tokens:

```python
from llamojo import Llama

# SmolLM2, TinyLlama, Qwen, etc. have their embedded chat templates loaded automatically:
llm = Llama("data/SmolLM2-135M-Instruct-Q4_K_M.gguf")

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a concise AI tutor."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=32,
    temperature=0.0,
)
print(response["choices"][0]["message"]["content"])
```

---

## 2. Explicit Built-in Chat Formats (`chat_format="..."`)
You can override or enforce a specific format using the `chat_format` argument:

```python
# Force Llama-3 formatting (<|start_header_id|>...<|end_header_id|>)
llm = Llama("data/model.gguf", chat_format="llama-3")

# Force Gemma formatting (<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n)
llm = Llama("data/model.gguf", chat_format="gemma")

# Force Llama-2 formatting (<s>[INST] <<SYS>>...<</SYS>>...[/INST])
llm = Llama("data/model.gguf", chat_format="llama-2")

# Force Mistral-Instruct formatting ([INST] ... [/INST])
llm = Llama("data/model.gguf", chat_format="mistral-instruct")

# Force standard ChatML formatting (<|im_start|>role\ncontent<|im_end|>)
llm = Llama("data/model.gguf", chat_format="chatml")
```

---

## 3. Streaming Chat Completion (`stream=True`)
Streaming outputs chunks conforming to OpenAI's streaming chat schema (`chat.completion.chunk`) with appropriate `delta`:

```python
stream = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    max_tokens=64,
    stream=True,
)

for chunk in stream:
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
print()
```

---

## 4. Custom Chat Formatter Registration
Users can register custom chat formatters matching `llama_cpp`'s decorator pattern:

```python
from llamojo import register_chat_format, ChatFormatterResponse

@register_chat_format("my_custom_format")
def format_custom(messages, **kwargs):
    prompt = "### Conversation:\n"
    for m in messages:
        prompt += f"{m['role'].upper()}: {m['content']}\n"
    prompt += "ASSISTANT: "
    return ChatFormatterResponse(prompt=prompt, stop=["\nUSER:"])

llm = Llama("data/model.gguf", chat_format="my_custom_format")
```

