// Vanta sanitizer proof: deliberately broken kernels whose defects are caught
// by Compute Sanitizer, contrasted with a correct kernel that stays clean.
//
// This is NOT a product kernel. It exists to prove the sanitizer workflow
// itself: a correct implementation passes memcheck/racecheck/synccheck, while
// each broken variant is flagged by the tool that targets its defect class.
//
// Modes:
//   sanitizer_proof correct       clean vector add
//   sanitizer_proof broken_oob    out-of-bounds global write   -> memcheck
//   sanitizer_proof broken_sync   divergent __syncthreads       -> synccheck
//   sanitizer_proof broken_race   unsynchronized shared memory  -> racecheck
//
// The Python checker (tools/check_sanitizer.py) owns the pass/fail decision.

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

__global__ void k_correct(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void k_oob(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Off-by-one write past the allocated output buffer.
        out[i + 1] = a[i] + b[i];
    }
}

__global__ void k_sync(const float* a, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Divergent __syncthreads: only even threads synchronize, so the
        // barrier is not reached by all threads in the block.
        if (threadIdx.x % 2 == 0) {
            __syncthreads();
        }
        out[i] = a[i];
        if (threadIdx.x % 2 == 1) {
            __syncthreads();
        }
    }
}

__global__ void k_race(const float* a, const float* b, float* out, int n) {
    __shared__ float shared_value;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        shared_value = a[0];
    }
    // Read the shared value without a barrier: a race between the writing
    // thread and every reading thread.
    const float x = shared_value;
    if (i < n) {
        out[i] = x + b[i];
    }
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int run(const char* mode) {
    const int n = 256;
    const int block = 128;
    const int grid = (n + block - 1) / block;

    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> out(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    if (!check_cuda(cudaMalloc(&device_a, n * sizeof(float)), "cudaMalloc(a)")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&device_b, n * sizeof(float)), "cudaMalloc(b)")) {
        cudaFree(device_a);
        return 1;
    }
    if (!check_cuda(cudaMalloc(&device_out, n * sizeof(float)),
                    "cudaMalloc(out)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        return 1;
    }
    if (!check_cuda(
            cudaMemcpy(device_a, a.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(a)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_out);
        return 1;
    }
    if (!check_cuda(
            cudaMemcpy(device_b, b.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(b)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_out);
        return 1;
    }

    const std::string selected(mode);
    if (selected == "correct") {
        k_correct<<<grid, block>>>(device_a, device_b, device_out, n);
    } else if (selected == "broken_oob") {
        k_oob<<<grid, block>>>(device_a, device_b, device_out, n);
    } else if (selected == "broken_sync") {
        k_sync<<<grid, block>>>(device_a, n, device_out);
    } else if (selected == "broken_race") {
        k_race<<<grid, block>>>(device_a, device_b, device_out, n);
    }

    const bool ok = check_cuda(cudaDeviceSynchronize(), "kernel");
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_out);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./sanitizer_proof correct|broken_oob|broken_sync|"
                     "broken_race\n";
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "correct" && mode != "broken_oob" && mode != "broken_sync" &&
        mode != "broken_race") {
        std::cerr << "Unknown mode: " << mode << '\n';
        return 2;
    }
    return run(mode.c_str());
}
