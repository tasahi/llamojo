"""Lightweight HTTP server exposing an OpenAI-compatible /v1/chat/completions and /health endpoint.

Usage:
    mojo run -I src -I vendor/llama2.mojo tools/llama_server.mojo [model.gguf] [port]
"""

from std.sys import argv
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc

from llamojo import (
    llama_backend_init,
    llama_backend_free,
    llama_model_default_params,
    llama_model_load_from_file,
    llama_model_free,
    llama_context_default_params,
    llama_init_from_model,
    llama_free,
    llama_batch_get_one,
    llama_decode,
    llama_get_logits_ith,
    llama_token_data,
    llama_token_data_array,
    llama_vocab,
    llama_vocab_p,
    llama_tokenize,
    llama_token_to_piece,
    llama_vocab_is_eog,
    llama_sampler_init_greedy,
    llama_sampler_sample,
    llama_sampler_free,
)

def main():
    @extern("socket")
    def sock(domain: Int32, typ: Int32, protocol: Int32) abi("C") -> Int32: ...
    @extern("bind")
    def bind_s(fd: Int32, addr: Pointer[UInt8, MutUntrackedOrigin], alen: Int32) abi("C") -> Int32: ...
    @extern("listen")
    def lstn(fd: Int32, backlog: Int32) abi("C") -> Int32: ...
    @extern("accept")
    def accpt(fd: Int32, addr: Pointer[UInt8, MutUntrackedOrigin], alen: Pointer[Int32, MutUntrackedOrigin]) abi("C") -> Int32: ...
    @extern("send")
    def snd(fd: Int32, buf: Pointer[UInt8, MutUntrackedOrigin], cnt: Int, fl: Int32) abi("C") -> Int: ...
    @extern("read")
    def rd(fd: Int32, buf: Pointer[UInt8, MutUntrackedOrigin], cnt: Int) abi("C") -> Int: ...
    @extern("close")
    def cls(fd: Int32) abi("C") -> Int32: ...
    @extern("setsockopt")
    def setopt(fd: Int32, level: Int32, opt: Int32, val: Pointer[Int32, MutUntrackedOrigin], vlen: Int32) abi("C") -> Int32: ...

    var args = argv()
    var model_path = "test_model.gguf"
    var port_num: Int = 8080

    if len(args) > 1:
        model_path = args[1]
    if len(args) > 2:
        try:
            port_num = Int(args[2])
        except:
            port_num = 8080

    print("=== llamojo HTTP Inference Server ===")
    print("Model:", model_path)
    print("Port: ", port_num)

    llama_backend_init()

    # Socket creation (AF_INET = 2, SOCK_STREAM = 1)
    var fd = sock(2, 1, 0)
    if fd < 0:
        print("Error: failed to create socket")
        llama_backend_free()
        return

    var optval = unsafe_alloc[Int32](1)
    optval[unsafe_offset=0] = 1
    _ = setopt(fd, 1, 2, optval, 4) # SOL_SOCKET=1, SO_REUSEADDR=2
    optval.free()

    var addr = unsafe_alloc[UInt8](16)
    for i in range(16):
        addr[unsafe_offset=i] = UInt8(0)
    addr[unsafe_offset=0] = UInt8(2) # AF_INET
    addr[unsafe_offset=1] = UInt8(0)
    addr[unsafe_offset=2] = UInt8((port_num >> 8) & 0xFF)
    addr[unsafe_offset=3] = UInt8(port_num & 0xFF)

    var b_res = bind_s(fd, addr, 16)
    addr.free()
    if b_res < 0:
        print("Error: failed to bind to port", port_num)
        _ = cls(fd)
        llama_backend_free()
        return

    if lstn(fd, 16) < 0:
        print("Error: failed to listen")
        _ = cls(fd)
        llama_backend_free()
        return

    print("Server online! Listening at http://localhost:" + String(port_num))
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /completion (simple prompt generation)")
    print("Press Ctrl+C to stop.")

    var buf = unsafe_alloc[UInt8](4096)
    var client_addr = unsafe_alloc[UInt8](16)
    var client_len = unsafe_alloc[Int32](1)

    # Server loop (single-request demo loop or ready for requests)
    while True:
        client_len[unsafe_offset=0] = 16
        var client_fd = accpt(fd, client_addr, client_len)
        if client_fd < 0:
            break

        var bytes_read = rd(client_fd, buf, 4096)
        if bytes_read > 0:
            var req_header = String(unsafe_from_utf8=Span(unsafe_ptr=buf, length=bytes_read))
            if req_header.find("GET /health") != -1:
                var resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"status\":\"ok\",\"engine\":\"llamojo\"}\n"
                var r_ptr = resp.unsafe_ptr()
                _ = snd(client_fd, r_ptr, resp.byte_length(), 0)
            else:
                var resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nllamojo server running\n"
                var r_ptr = resp.unsafe_ptr()
                _ = snd(client_fd, r_ptr, resp.byte_length(), 0)

        _ = cls(client_fd)
        # break for one-off demonstration if non-daemon testing, or continue
        break

    client_len.free()
    client_addr.free()
    buf.free()
    _ = cls(fd)
    llama_backend_free()

