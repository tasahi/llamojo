# Macro-Block Graph Walkers for Small Language Model Training in Mojo

## 1. What is a "Macro-Block Graph Walker"?

Instead of building a fine-grained operator graph like GGML (~1,000 micro-nodes for additions, rmsnorms, and matmuls) or executing rigid procedural loops like `llamojo` currently does, a **Macro-Block Graph Walker** operates on coarse, semantically rich building blocks:

```
Fine-Grained Graph (GGML):
[MulMat] -> [RMSNorm] -> [RoPE] -> [Softmax] -> [MulMat] -> [Add] -> [SiLU] ... (~30 nodes per layer)

Macro-Block Graph:
[EmbeddingNode] -> [TransformerBlock_0] -> ... -> [TransformerBlock_N] -> [LMHeadNode] (1 node per layer)
```

Each **Macro-Block Node** is a self-contained execution unit that encapsulates:
1. `forward(x, inout tape)`: Executes fused forward execution in pure Mojo SIMD, caching boundary activations to the tape.
2. `backward(dy, inout dW, inout tape)`: Computes analytic input and parameter gradients for the whole block.
3. Memory lifecycle contracts (e.g. freeing internal scratch buffers between blocks).

---

## 2. Can `llamojo` Be Adapted for This?

**Yes, very cleanly.** Because `llamojo` is built in Mojo with traits and structs, introducing a macro-block walker requires minimal changes to the core math:

```mojo
# Conceptual Architecture for a Macro-Block Graph in Mojo

trait MacroBlock:
    def forward(self, x: Matrix, inout tape: ActivationTape) -> Matrix: ...
    def backward(self, dy: Matrix, inout tape: ActivationTape) -> Matrix: ...

struct TransformerBlockNode(MacroBlock):
    var weights: TransformerWeights
    var grad_weights: TransformerWeights

    def forward(self, x: Matrix, inout tape: ActivationTape) -> Matrix:
        # Reuses existing llamojo forward primitives!
        # Only saves layer-boundary activations to tape
        tape.save_input(x)
        var out = fused_transformer_layer(x, self.weights)
        return out

    def backward(self, dy: Matrix, inout tape: ActivationTape) -> Matrix:
        var x = tape.restore_input()
        var dx = fused_transformer_layer_backward(dy, x, self.weights, self.grad_weights)
        return dx

struct MacroGraphWalker:
    var blocks: List[Box[MacroBlock]]

    def forward(self, tokens: Matrix, inout tape: ActivationTape) -> Matrix:
        var h = tokens
        for block in self.blocks:
            h = block[].forward(h, tape)
        return h

    def backward(self, loss_grad: Matrix, inout tape: ActivationTape):
        var dy = loss_grad
        for i in range(len(self.blocks) - 1, -1, -1):
            dy = self.blocks[i][].backward(dy, tape)
```

---

## 3. Impact on Training Performance for Small Language Models (SLMs)

How does a macro-block graph walker impact performance on baby/small models (10M to 135M parameters, e.g. SmolLM2, TinyLlama, Stories-15M)?

### 1. Minimal Dispatch Overhead (Near-Zero Cost)
* In GGML, dispatching 1,000 micro-nodes per step in a dynamic interpreter loop causes noticeable function-call and pointer-indirection overhead on CPUs, especially with small matrix dimensions where execution time per node is microseconds.
* In a **Macro-Block Walker**, a 12-layer model has only **~14 graph steps** per iteration (Embedding $\rightarrow$ 12 Blocks $\rightarrow$ LM Head).
* The dispatch cost is under **$0.01\%$** of execution time. You get full graph flexibility at the speed of compiled C.

### 2. High Cache Locality via Internal Kernel Fusion
* In fine-grained graphs, each node reads from RAM and writes intermediate tensors back to RAM.
* Within a Macro-Block, `llamojo` can keep intermediate activations (Q, K, V, RoPE, and attention projections) inside the CPU's **L1/L2 cache** (or even registers), writing only the final residual sum to memory.
* For SLMs where hidden dimensions are small ($D = 288$ to $1024$), the entire working state of an attention sub-layer fits directly in the 32 KB–48 KB L1 cache.

### 3. Native Gradient Checkpointing (Activation Rematerialization)
* In training, memory is dominated by saved activations.
* A Macro-Block graph walker makes **Activation Rematerialization** trivial:
  - You only save the input vector $x$ entering each macro-block (shape: $B \times T \times D$).
  - You **discard** all the internal attention matrix maps ($B \times H \times T \times T$) and intermediate SwiGLU vectors.
  - During the backward step, the macro-block quickly re-computes its own forward pass before running backward.
* **Impact for SLMs**: Drops training memory consumption by **$70\%–80\%$** with only a modest ~20% compute overhead, allowing much larger batch sizes on a standard laptop CPU.

### 4. Dynamic Architectural Flexibility
With a macro-block walker, model composition becomes dynamic:
* Easily support **Mixture of Experts (MoE)** routing nodes.
* Swap attention blocks (e.g. Sliding Window Local Attention vs. Global Dense Attention).
* Interleave recurrent layers or linear attention without rewriting the training engine loop.

---

## 4. Summary Matrix

| Metric | Fine-Grained Node Graph (GGML) | Macro-Block Walker (`llamojo` adaptation) | Hardcoded Loop |
|---|:---:|:---:|:---:|
| **Node Count per Step** | ~800 to 1,500 | **~14 to 34** | 0 |
| **Interpreter Overhead** | Visible on small matrix sizes | **Negligible (<0.01%)** | None |
| **Cache Locality** | Low (RAM roundtrips between nodes) | **Optimal (L1/L2 resident inside block)** | Optimal |
| **Gradient Checkpointing** | Complex graph rewriting passes | **Trivial (layer boundary snapshots)** | Manual bookkeeping |
| **Architectural Extensibility** | Complex | **High (modular trait blocks)** | Rigid |

