// Sanitizer proof kernels for the Vanta GPU validation lab.
//
// This file is not a product kernel. It proves the sanitizer workflow: a
// `correct` mode passes memcheck, racecheck, and synccheck, while three
// deliberately broken modes are each caught by the tool that targets their
// defect class:
//
//   correct      clean baseline, passes memcheck + racecheck + synccheck
//   broken_oob   one thread writes one element past the allocation -> memcheck
//   broken_sync  one thread hits __syncthreads alone (divergent barrier) -> synccheck
//   broken_race  one thread writes a shared variable, others read it without
//                synchronization                                   -> racecheck
//
// Run with a single argument: the mode name.
// Optional second argument is a tool hint (memcheck/racecheck/synccheck) that
// only selects which kernel to launch; the verdict is decided by
// tools/check_sanitizer.py against the actual sanitizer output.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef N
#define N 256
#endif
#ifndef BLOCK
#define BLOCK 128
#endif

__global__ void k_correct(int *g) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    g[i] = i;
}

__global__ void k_oob(int *g) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    // One out-of-bounds write: thread 0 stamps one element past the
    // allocation. memcheck must flag it.
    if (threadIdx.x == 0) {
        g[N] = -1;
    }
    g[i] = i;
}

__global__ void k_sync(int *g) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    // Divergent barriers: the lower and upper halves of the block take
    // different __syncthreads() calls, so some threads in the block skip each
    // barrier. synccheck must flag both as unsynchronized barriers.
    __shared__ int s;
    if (threadIdx.x < BLOCK / 2) {
        __syncthreads();
    }
    s = i;
    if (threadIdx.x >= BLOCK / 2) {
        __syncthreads();
    }
    g[i] = i + s;
}

__global__ void k_race(int *g) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    // Each thread writes its own shared slot, then reads a neighbor's slot
    // that another thread just wrote, with no __syncthreads between the write
    // and the read. This is a cross-thread RAW race on shared memory that
    // racecheck must flag.
    __shared__ int s[BLOCK];
    s[threadIdx.x] = threadIdx.x * 2;
    int v = s[(threadIdx.x + 1) % BLOCK];
    g[i] = i + v;
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "correct";
    (void)(argc > 2 ? argv[2] : nullptr);  // tool hint, unused by the kernel

    int *d = nullptr;
    if (cudaMalloc(&d, N * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 2;
    }

    // Single block so every broken-mode defect is exercised in one launch and
    // the divergent/unsynchronized constructs are well defined within the block.
    int grid = 1;
    int threads = BLOCK;

    if (strcmp(mode, "correct") == 0) {
        k_correct<<<grid, threads>>>(d);
    } else if (strcmp(mode, "broken_oob") == 0) {
        k_oob<<<grid, threads>>>(d);
    } else if (strcmp(mode, "broken_sync") == 0) {
        k_sync<<<grid, threads>>>(d);
    } else if (strcmp(mode, "broken_race") == 0) {
        k_race<<<grid, threads>>>(d);
    } else {
        fprintf(stderr, "unknown mode: %s (expected correct, broken_oob, "
                        "broken_sync, or broken_race)\n", mode);
        cudaFree(d);
        return 2;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        cudaFree(d);
        return 2;
    }
    cudaFree(d);
    return 0;
}
