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
    g[i] = i;
    // Divergent barrier: only thread 0 reaches __syncthreads while the rest of
    // the block continue. synccheck must flag the unsynchronized barrier.
    if (threadIdx.x == 0) {
        __syncthreads();
    }
    g[i] = g[i] + 1;
}

__global__ void k_race(int *g) {
    __shared__ int shared_value;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    // Thread 0 writes shared_value; every other thread reads it with no
    // __syncthreads between the write and the read. racecheck must flag the
    // unsynchronized shared-memory hazard.
    if (threadIdx.x == 0) {
        shared_value = 7;
    }
    int v = shared_value;
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

    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
