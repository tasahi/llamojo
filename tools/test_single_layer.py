import numpy as np
from llama_cpp import Llama

llm = Llama(
    model_path='$MODEL_PATH',
    n_ctx=128,
    logits_all=True,
    verbose=False
)
llm.reset()
llm.eval([2])
sc = np.array(llm.scores[0])
print("llama.cpp after tok 2:")
top5 = np.argsort(sc)[-5:][::-1]
for t in top5:
    print(f"  {t} ({sc[t]:.4f}) {repr(llm.detokenize([int(t)]).decode('utf-8', errors='replace'))}")
