import numpy as np

# In llama.cpp:
# n_embd_per_layer = 256
# n_layer = 35
# n_tokens = 1
#
# inp_batch shape: [1536, 1]
# per_layer_model_proj shape in GGUF: [1536, 8960] (ne = [1536, 8960])
# per_layer_proj = ggml_mul_mat(per_layer_model_proj, inp_batch) -> shape [8960, 1]
# per_layer_proj = reshape_3d(per_layer_proj, 256, 35, 1) -> ne[0]=256, ne[1]=35, ne[2]=1
# build_norm: RMSNorm over ne[0]=256 for each (ne[1], ne[2]) slice!
#
# build_inp_per_layer:
# inp_per_layer = get_rows(per_layer_tok_embd, tokens)
# per_layer_tok_embd shape in GGUF: [8960, 262144] (ne = [8960, 262144])
# get_rows -> ne = [8960, 1]
# inp_per_layer = reshape_3d(inp_per_layer, 256, 35, 1) -> ne[0]=256, ne[1]=35, ne[2]=1
#
# inp_per_layer = add(per_layer_proj, inp_per_layer) -> ne = [256, 35, 1]
# permute(0, 2, 1, 3):
# ne[0]=256, ne[1]=1, ne[2]=35!
#
# gemma4_view_2d_slice(ctx0, inp_per_layer, il):
# idx = il (the 3rd dim, i.e. ne[2])!
# view_2d: offset = idx * ne[0] * ne[1] * sizeof(float)
# = il * 256 * 1 * sizeof(float) = il * 256 * sizeof(float)!
#
# But wait! In the permuted tensor:
# The original tensor had shape [256, 35, 1].
# In ggml column-major storage:
# original element (i, l, t) has offset: i + l * 256 + t * (256 * 35).
# After permute(0, 2, 1, 3):
# ggml_cont copies the permuted tensor into a contiguous buffer of shape [256, n_tokens, 35]!
# Its strides are:
# dim 0 (size 256): stride 1
# dim 1 (size n_tokens=1): stride 256
# dim 2 (size 35): stride 256 * n_tokens!
# For n_tokens = 1:
# dim 2 stride is 256 * 1 = 256!
# So slice for layer il is at linear offset il * 256!
print("For n_tokens=1, linear offset of layer il is EXACTLY il * 256!")
