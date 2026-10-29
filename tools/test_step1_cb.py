import ctypes
from llama_cpp import llama_cpp

cb_type = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)

def callback(t_ptr, ask, user_data):
    if not t_ptr or ask:
        return True
    class GGMLTensor(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("backend", ctypes.c_int),
            ("buffer", ctypes.c_void_p),
            ("ne", ctypes.c_int64 * 4),
            ("nb", ctypes.c_size_t * 4),
            ("op", ctypes.c_int),
            ("op_params", ctypes.c_int32 * 16),
            ("flags", ctypes.c_int32),
            ("src", ctypes.c_void_p * 10),
            ("view_src", ctypes.c_void_p),
            ("view_offs", ctypes.c_size_t),
            ("data", ctypes.c_void_p),
            ("name", ctypes.c_char * 64),
            ("extra", ctypes.c_void_p),
            ("padding", ctypes.c_char * 8),
        ]
    t = GGMLTensor.from_address(t_ptr)
    name = t.name.decode('utf-8', errors='replace')
    if name in ["Qcur_pos-0", "Kcur_pos-0", "kqv_out-0"]:
        if t.data:
            arr = (ctypes.c_float * 4).from_address(t.data)
            print(f"[{name}] {[round(x, 4) for x in arr]}")
    return True

c_callback = cb_type(callback)

mparams = llama_cpp.llama_model_default_params()
model = llama_cpp.llama_model_load_from_file(
    b"$MODEL_PATH",
    mparams
)

cparams = llama_cpp.llama_context_default_params()
cparams.n_ctx = 128
cparams.cb_eval = c_callback

ctx = llama_cpp.llama_init_from_model(model, cparams)

# Eval pos 0
b0 = llama_cpp.llama_batch_init(1, 0, 1)
b0.token[0] = 2
b0.pos[0] = 0
b0.logits[0] = 0
b0.n_tokens = 1
llama_cpp.llama_decode(ctx, b0)

print("--- Decoding pos 1 ---")
# Eval pos 1
b1 = llama_cpp.llama_batch_init(1, 0, 1)
b1.token[0] = 14946
b1.pos[0] = 1
b1.logits[0] = 1
b1.n_tokens = 1
llama_cpp.llama_decode(ctx, b1)

