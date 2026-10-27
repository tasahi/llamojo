from llama_cpp import Llama
import numpy as np

llm = Llama(
    model_path='/home/tasahi/.cache/huggingface/hub/blobs/a1/a156058e62f4541ff0b9226ed3bc57fdb30ddc9d266f63cc1716a3eb32ab955c',
    n_ctx=128,
    logits_all=True,
    verbose=False
)
# Prompt: "Once upon a time"
prompt_tokens = [2, 14946, 3324, 496, 990]

# In llama.cpp, what happens when we eval all 5 tokens together vs one by one?
llm.reset()
llm.eval(prompt_tokens)
print("After batch eval prompt_tokens:")
for p in range(5):
    top_t = llm.scores[p].argmax()
    print(f"  pos {p}: top={top_t} ({llm.scores[p][top_t]:.3f})")

