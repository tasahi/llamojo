#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

// Minimal C declarations matching llama.h
typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

typedef struct llama_model* llama_model_p;
typedef struct llama_context* llama_context_p;
typedef struct llama_vocab* llama_vocab_p;
typedef struct llama_sampler* llama_sampler_p;

// C-ABI function prototypes from libllama.so
void llama_backend_init(void);
void llama_backend_free(void);
int64_t llama_time_us(void);
int64_t llama_max_devices(void);
bool llama_supports_mmap(void);
bool llama_supports_mlock(void);
bool llama_supports_gpu_offload(void);

llama_sampler_p llama_sampler_init_greedy(void);
void llama_sampler_free(llama_sampler_p smpl);

int main(void) {
    printf("=== Testing llamojo libllama.so C-ABI ===\n");

    llama_backend_init();

    int64_t t0 = llama_time_us();
    printf("  llama_time_us: %ld\n", (long)t0);
    assert(t0 > 0);

    int64_t devs = llama_max_devices();
    printf("  llama_max_devices: %ld\n", (long)devs);
    assert(devs >= 1);

    bool mmap_ok = llama_supports_mmap();
    bool mlock_ok = llama_supports_mlock();
    bool gpu_ok = llama_supports_gpu_offload();
    printf("  supports_mmap: %d, supports_mlock: %d, gpu_offload: %d\n", mmap_ok, mlock_ok, gpu_ok);
    assert(mmap_ok == true);

    llama_sampler_p smpl = llama_sampler_init_greedy();
    assert(smpl != NULL);
    printf("  llama_sampler_init_greedy created opaque handle: %p\n", (void*)smpl);
    llama_sampler_free(smpl);

    llama_backend_free();
    printf("=== C-ABI integration test passed! ✓ ===\n");
    return 0;
}

