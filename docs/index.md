---
title: "llamojo Documentation Bundle"
version: "0.2.0"
type: "index"
description: "Central technical documentation index for llamojo: pure Mojo low-resource CPU inference engine, GGUF model runner, tokenizer implementations, and Python C-ABI bindings."
entries:
  - implementation_guide.md
  - pure_cpu_implementation_plan.md
  - pure_cpu_extension_plan.md
  - gap_analysis_llama2_c.md
  - llama_cpp_pure_mojo_study.md
  - ggml_relationship.md
  - dual_library_architecture.md
  - gojo_llamojo_consistency_analysis.md
  - ggojo_integration_gaps.md
  - ggojo_macro_graph_integration.md
  - gpt2_encoder_architecture_and_importance.md
  - mmbert_gemma2_tokenizer_feasibility.md
  - modernbert_laya_inference_plan.md
  - whisper_integration_study.md
  - macro_block_graph_training.md
  - training_gap_analysis.md
  - framework_standards_comparison.md
  - interop_and_pluggable_backends_plan.md
  - pytorch_interop_plan.md
  - hf_compat_facade.md
  - python_gap_analysis.md
  - python_from_pretrained.md
  - python_chat_templates.md
  - python_logprobs.md
  - python_logit_processors.md
  - python_kv_cache_serialization.md
  - phases/index.md
tags: [llamojo, mojo, inference-engine, gguf, cpu-inference, simd, bpe-tokenizer, c-abi, python-bindings]
timestamp: 2026-11-05T00:00:00Z
---

# `llamojo` Technical Knowledge Base

Welcome to the **`llamojo`** technical documentation bundle, structured according to the [Open Knowledge Format (OKF)](https://okf.md/quickstart/).

`llamojo` is a pure Mojo, standalone, high-performance CPU inference engine for Large and Small Language Models. It directly parses standard GGUF binary models, implements fused SIMD quantization kernels (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`), manages zero-overhead linear memory arenas and KV caches, and exposes an idiomatic Python interface compatible with `llama-cpp-python` and Hugging Face pipelines.

---

## 🧭 Technical Summaries & Documentation Map

### 1. Engine Core & Architecture Foundations
Core system design, memory management, GGUF parsing, transformer forward passes, and comparative architecture studies.

| Document | Type | Status | Technical Focus / Summary |
|---|---|---|---|
| [**System Implementation Guide**](./implementation_guide.md) | `reference` | Completed | Primary architectural blueprint detailing linear arenas, GGUF mmap parsing, KV cache indexing, and autoregressive generation. |
| [**Pure-CPU Implementation Plan**](./pure_cpu_implementation_plan.md) | `howto` | Completed | Roadmap for prompt ingestion, token generation loops, and initial GGUF compliance. |
| [**Pure-CPU Extension Roadmap (Phases 5–11)**](./pure_cpu_extension_plan.md) | `howto` | Completed | Comprehensive engineering roadmap covering advanced quantization (`Q4_K`, `Q6_K`), thread pooling, and MoE routing. |
| [**Gap Analysis vs. llama2.c / llama2.mojo**](./gap_analysis_llama2_c.md) | `architecture-decision` | Completed | Detailed comparison highlighting memory layout, zero-copy tensor slicing, and quantization advantages over basic ports. |
| [**libllama / llama.cpp Architectural Study**](./llama_cpp_pure_mojo_study.md) | `concept` | Completed | In-depth analysis of upstream `llama.cpp` mechanisms and the feasibility of pure-Mojo kernel re-implementations. |
| [**GGML Relationship & Emulation Study**](./ggml_relationship.md) | `concept` | Completed | Deep dive clarifying why `llamojo` runs pure Mojo without linking or depending on external C++ GGML binaries. |

---

### 2. Cross-Library Synergy: `llamojo` & `ggojo` (Engine & Training)
The modular separation between **`llamojo`** (inference runtime, tokenizers, KV cache) and **`ggojo`** (tensor math, Macro-Block graphs, autodiff, and training).

| Document | Type | Status | Technical Focus / Summary |
|---|---|---|---|
| [**Dual-Library Architecture**](./dual_library_architecture.md) | `concept` | Completed | Strategic separation of concerns: zero-dependency inference (`llamojo`) vs. gradient training engine (`ggojo` / `gradojo`). |
| [**Consistency Analysis: ggojo vs. llamojo**](./gojo_llamojo_consistency_analysis.md) | `concept` | Completed | Byte-level alignment audit of struct definitions, tensor descriptors, and pointer conventions. |
| [**Analysis of Pending Integration Gaps**](./ggojo_integration_gaps.md) | `architecture-decision` | Completed | Catalog of pending integration boundaries: tensor view wrappers, graph dispatch adapters, and scratch arenas. |
| [**Macro-Block Graph Integration (Gaps 1 & 5)**](./ggojo_macro_graph_integration.md) | `howto` | Completed | Operational guide for executing `ggojo` coarse-grained Macro-Blocks within `llamojo` decoding pipelines. |

---

### 3. Tokenizer Subsystems, Encoders & Speech / Audio (Whisper ASR)
Detailed architectural studies of tokenization algorithms (Byte-level BPE, SentencePiece), bidirectional encoder backbones, and encoder-decoder ASR models.

| Document | Type | Status | Technical Focus / Summary |
|---|---|---|---|
| [**GPT-2 / LLaMA-3 BPE Deep Dive**](./gpt2_encoder_architecture_and_importance.md) | `concept` | Completed | Mathematical foundation and implementation of regex-based byte-level BPE with merge caching. |
| [**mmBERT / Gemma 2 Tokenizer Feasibility**](./mmbert_gemma2_tokenizer_feasibility.md) | `concept` | Completed | Engineering study for supporting 256,016-vocabulary multilingual SentencePiece tokenizers in pure Mojo. |
| [**ModernBERT & Laya Inference Engine Plan**](./modernbert_laya_inference_plan.md) | `howto` | Completed | Inference specification for bidirectional attention, alternating sliding windows, and Laya decision scoring heads. |
| [**Whisper ASR Integration Study**](./whisper_integration_study.md) | `concept` | Active | Feasibility study of `whisper.cpp` architecture, component reuse in `llamojo`, and comparison of integrated vs. standalone implementations. |

---

### 4. Training Foundations & Autodiff Integration
Foundational research into CPU-native deep learning training, activation tapes, and framework parity.

| Document | Type | Status | Technical Focus / Summary |
|---|---|---|---|
| [**Architecture & Implementation Gap Analysis: Training**](./training_gap_analysis.md) | `architecture-decision` | Completed | Analysis of autodiff requirements, backward passes, gradient checkpointing, and AdamW optimizer states. |
| [**Macro-Block Graph Walkers for Training**](./macro_block_graph_training.md) | `concept` | Completed | Fused layer-boundary activation checkpointing slashing CPU training memory footprint by over 90%. |
| [**Framework Standards Comparison**](./framework_standards_comparison.md) | `concept` | Completed | Paradigm comparison between PyTorch, JAX, GGML, and Mojo native execution patterns. |

---

### 5. Interoperability, PyTorch & Python Bindings
Bridging native Mojo compute with Python runtimes, Hugging Face Hub, PyTorch tensors, and standard chat templates.

| Document | Type | Status | Technical Focus / Summary |
|---|---|---|---|
| [**Pluggable Compute & Interoperability Architecture**](./interop_and_pluggable_backends_plan.md) | `concept` | Active | Zero-copy tensor delegation between `llamojo`, `ggojo`, NuMojo, and MAX Graph engines. |
| [**PyTorch Interoperability Architecture (Inference)**](./pytorch_interop_plan.md) | `reference` | Completed | Specification for PyTorch tensor exchange, HuggingFace pipeline integration, and DLPack zero-copy pointers. |
| [**Hugging Face / PyTorch-Style Facade (`hf_compat`)**](./hf_compat_facade.md) | `reference` | Completed | Python facade providing `AutoModelForCausalLM` and `AutoTokenizer` APIs powered by `libllamojo.so`. |
| [**Functional Gap Analysis vs. llama-cpp-python**](./python_gap_analysis.md) | `architecture-decision` | Completed | Comprehensive parity audit across methods, kwargs, and generation flags. |
| [**Hugging Face Hub Integration (`from_pretrained`)**](./python_from_pretrained.md) | `reference` | Completed | Native downloading, caching, and loading of GGUF models directly from Hugging Face Hub repositories. |
| [**Chat Template Engines**](./python_chat_templates.md) | `reference` | Completed | Jinja2-style chat templating support (ChatML, LLaMA-3, Mistral, Gemma) across Python and Mojo. |
| [**Token Logprob Computation**](./python_logprobs.md) | `reference` | Completed | Extraction of top-k token log-probabilities formatted to OpenAI API schemas. |
| [**Logit Processors & Biases**](./python_logit_processors.md) | `reference` | Completed | Hook infrastructure for custom logit biases, repetition penalties, and steering vectors. |
| [**KV Cache & Context State Serialization**](./python_kv_cache_serialization.md) | `reference` | Completed | Context saving and resumption (`save_state` / `load_state`) preserving prompt caches across sessions. |

---

## 🔬 Granular Implementation Phases

All detailed engineering execution phases, benchmark reports, and step-by-step verification milestones are organized in the [`docs/phases/`](./phases/index.md) sub-bundle:

- 📑 [**Browse All Implementation Phases (`docs/phases/index.md`)**](./phases/index.md)
  - **Gemma 4 Architecture (Phases 1–5)**: PLE, per-head RMSNorm, GeGLU, alternating SWA/global attention.
  - **SIMD Kernel Acceleration (Phases 1–7)**: Unrolled SIMD loops, thread pooling, cacheline alignment, activation quantization to Q8, 2-row register tiling, AVX2 intrinsics.
  - **Python C-ABI Shared Library (Phases 1–3)**: `libllamojo.so`, ctypes bindings, streaming generators, pip packaging.
  - **Grammar-Constrained Decoding & LoRA (Steps 1–5)**: GBNF compiler, token logits masking, and runtime dynamic LoRA swapping.

---

## 🏛️ Repository Architecture

```
llamojo/
├── docs/                                     # OKF Knowledge Bundle (this directory)
│   ├── index.md                              # Root OKF index and technical summaries
│   └── phases/                               # Granular engineering implementation phases
│       ├── index.md                          # Phase sub-bundle index
│       ├── gemma4_*.md                       # Gemma 4 architecture & SIMD acceleration phases
│       ├── python_integration_*.md           # libllamojo.so C-ABI & Python bindings phases
│       └── python_grammar_lora_*.md          # GBNF grammar & LoRA adapter phases
├── src/
│   └── llamojo/
│       ├── core/                             # Linear allocators, tensor views, config schemas
│       ├── model/                            # Transformer forward graph, layers, and blocks
│       ├── kernels/                          # SIMD quantization dot products (Q4_0, Q8_0, Q4_K, Q6_K)
│       ├── vocab/                            # Byte-level BPE, SentencePiece, Gemma 2 tokenizers
│       ├── sampling/                         # Top-p, top-k, temperature, min-p, GBNF masking
│       └── ffi/                              # C-ABI export symbols (libllamojo.so)
├── python/
│   └── llamojo/                              # Python package (Llama, AutoModelForCausalLM, AutoTokenizer)
├── tests/                                    # Automated Mojo & Python test suites
└── scripts/                                  # Build and benchmarking scripts
```

