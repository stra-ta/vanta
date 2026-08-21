// Vanta matmul proof: tiled square matrix multiplication validation.
//
// This is the third Vanta proof. It multiplies two N x N float32 matrices with
// a tiled shared-memory kernel and validates the result against a CPU
// double-precision reference.
//
// Modes:
//   matmul_proof correct
//   matmul_proof faulty
//   matmul_proof replay      (reads one matmul failure case from stdin)
//
// Replay line protocol:
//   <implementation>
//   <n>
//   <block_threads>
//   <a0> <a1> ... <a_{n*n-1}>
//   <b0> <b1> ... <b_{n*n-1}>
//
// The correctness tolerance is operation-aware. A tiled matmul accumulates N
// products per output element; the worst-case float32 rounding over N serial
// additions is bounded by N * epsilon * |result|. The C++ campaign gate and
// the Python replay classifier use the same bound (a small 3x safety margin),
// so the correct kernel passes and a structural defect fails by a wide margin.
//
// The deliberate defect drops the final inner product term (k == N - 1), so
// every output element misses A[i][N-1] * B[N-1][j]. This is a logical error,
// not an out-of-bounds access, so Compute Sanitizer stays clean.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int TILE = 16;

__global__ void matmul_tiled(const float* a, const float* b, float* c, int n,
                             bool faulty) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int tiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        as[ty][tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        bs[ty][tx] = (col < n && b_row < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            const int global_k = t * TILE + k;
            if (!faulty || global_k < n - 1) {
                acc += as[ty][k] * bs[k][tx];
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}

std::vector<float> make_mat(int n, int pattern) {
    std::vector<float> values(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float value = 1.0f;
            if (pattern == 0) {
                value = 1.0f;
            } else if (pattern == 1) {
                value = static_cast<float>((i % 2 == 0) ? 1 : -1);
            } else {
                value = static_cast<float>(((i * 7 + j * 3) % 9) + 1);
            }
            values[i * n + j] = value;
        }
    }
    return values;
}

void reference_matmul(const std::vector<float>& a,
                      const std::vector<float>& b, int n,
                      std::vector<double>& out) {
    out.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double total = 0.0;
            for (int k = 0; k < n; ++k) {
                total += static_cast<double>(a[i * n + k]) *
                         static_cast<double>(b[k * n + j]);
            }
            out[i * n + j] = total;
        }
    }
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

double matmul_tolerance_for(double scale, int k) {
    const double eps = std::ldexp(1.0, -23);  // float32 machine epsilon
    return std::max(1.0, std::abs(scale)) * k * 3.0 * eps;
}

bool run_matmul(const std::vector<float>& a, const std::vector<float>& b,
                int n, bool faulty, const cudaDeviceProp& device,
                std::vector<float>& out_host) {
    out_host.assign(static_cast<size_t>(n) * n, 0.0f);
    const int elements = n * n;
    if (n <= 0) {
        return false;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    bool ok = true;

    if (!check_cuda(cudaMalloc(&device_a, elements * sizeof(float)),
                    "cudaMalloc(a)")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&device_b, elements * sizeof(float)),
                    "cudaMalloc(b)")) {
        cudaFree(device_a);
        return false;
    }
    if (!check_cuda(cudaMalloc(&device_c, elements * sizeof(float)),
                    "cudaMalloc(c)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        return false;
    }

    if (!check_cuda(
            cudaMemcpy(device_a, a.data(), elements * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(a)")) {
        ok = false;
        goto cleanup;
    }
    if (!check_cuda(
            cudaMemcpy(device_b, b.data(), elements * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(b)")) {
        ok = false;
        goto cleanup;
    }

    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    matmul_tiled<<<blocks, threads>>>(device_a, device_b, device_c, n, faulty);
    if (!check_cuda(cudaDeviceSynchronize(), "matmul_tiled")) {
        ok = false;
        goto cleanup;
    }

    if (!check_cuda(
            cudaMemcpy(out_host.data(), device_c, elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(c)")) {
        ok = false;
        goto cleanup;
    }

cleanup:
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    (void)device;
    return ok;
}

void write_failure(const std::vector<float>& a, const std::vector<float>& b,
                   int n, double scale, double max_error,
                   const cudaDeviceProp& device) {
    std::ofstream file("failure.json");
    file << std::setprecision(9);
    file << "{\n";
    file << "  \"schema\": \"vanta.failure.v0\",\n";
    file << "  \"kernel\": \"matmul\",\n";
    file << "  \"implementation\": \"faulty_tile\",\n";
    file << "  \"input\": {\n";
    file << "    \"dtype\": \"float32\",\n";
    file << "    \"shape\": [" << n << ", " << n << ", " << n << "],\n";
    file << "    \"a\": [";
    for (int i = 0; i < n * n; ++i) {
        if (i > 0) {
            file << ", ";
        }
        file << a[i];
    }
    file << "],\n";
    file << "    \"b\": [";
    for (int i = 0; i < n * n; ++i) {
        if (i > 0) {
            file << ", ";
        }
        file << b[i];
    }
    file << "]\n";
    file << "  },\n";
    file << "  \"meta\": {\"n\": " << n << ", \"k\": " << n << "},\n";
    file << "  \"launch\": {\n";
    file << "    \"block_threads\": " << TILE << ",\n";
    file << "    \"grid_blocks\": 1\n";
    file << "  },\n";
    file << "  \"environment\": {\n";
    file << "    \"gpu\": \"" << device.name << "\",\n";
    file << "    \"compute_capability\": \"" << device.major << "."
         << device.minor << "\"\n";
    file << "  },\n";
    file << "  \"observed\": {\n";
    file << "    \"kind\": \"correctness_mismatch\",\n";
    file << "    \"expected\": " << scale << ",\n";
    file << "    \"actual\": " << max_error << "\n";
    file << "  }\n";
    file << "}\n";
}

int run_campaign(bool faulty) {
    cudaDeviceProp device{};
    if (!check_cuda(cudaGetDeviceProperties(&device, 0),
                    "cudaGetDeviceProperties")) {
        return 1;
    }

    std::cout << "GPU: " << device.name << '\n';
    std::cout << "Compute capability: " << device.major << '.'
              << device.minor << '\n';

    const std::vector<int> sizes = {16, 24, 32, 64, 128};
    const std::vector<int> patterns = {0, 1, 2};

    int pass_count = 0;
    int mismatch_count = 0;

    for (int n : sizes) {
        for (int pattern : patterns) {
            std::vector<float> a = make_mat(n, pattern);
            std::vector<float> b = make_mat(n, pattern + 1);
            std::vector<double> ref;
            reference_matmul(a, b, n, ref);
            double scale = 0.0;
            for (double value : ref) {
                scale = std::max(scale, std::abs(value));
            }

            std::vector<float> actual;
            if (!run_matmul(a, b, n, faulty, device, actual)) {
                return 1;
            }

            double max_error = 0.0;
            for (int i = 0; i < n * n; ++i) {
                max_error = std::max(
                    max_error, std::abs(static_cast<double>(actual[i]) -
                                        ref[i]));
            }
            const bool passed =
                max_error <= matmul_tolerance_for(scale, n);

            if (passed) {
                ++pass_count;
                std::cout << "PASS n=" << n << " pattern=" << pattern
                          << " max_error=" << max_error << '\n';
            } else {
                ++mismatch_count;
                std::cout << "MISMATCH n=" << n << " pattern=" << pattern
                          << " scale=" << scale << " max_error=" << max_error
                          << '\n';
                if (faulty && mismatch_count == 1) {
                    write_failure(a, b, n, scale, max_error, device);
                }
            }
        }
    }

    std::cout << "pass_count=" << pass_count << '\n';
    std::cout << "mismatch_count=" << mismatch_count << '\n';

    if (!faulty && mismatch_count == 0) {
        std::cout << "correct kernel: PASS\n";
        return 0;
    }
    if (faulty && mismatch_count > 0) {
        std::cout << "faulty kernel correctly detected: PASS\n";
        return 0;
    }
    std::cerr << "unexpected validation result\n";
    return 1;
}

int run_replay() {
    std::string implementation;
    int n = 0;
    int block_threads = 0;

    if (!(std::cin >> implementation >> n >> block_threads)) {
        std::cerr << "Invalid replay header\n";
        return 2;
    }
    if (implementation != "faulty_tile" && implementation != "correct") {
        std::cerr << "Unsupported implementation: " << implementation << '\n';
        return 2;
    }
    if (n <= 0) {
        std::cerr << "Invalid matrix size\n";
        return 2;
    }

    std::vector<float> a(static_cast<size_t>(n) * n);
    std::vector<float> b(static_cast<size_t>(n) * n);
    for (float& value : a) {
        if (!(std::cin >> value)) {
            std::cerr << "Not enough matrix A values\n";
            return 2;
        }
    }
    for (float& value : b) {
        if (!(std::cin >> value)) {
            std::cerr << "Not enough matrix B values\n";
            return 2;
        }
    }

    cudaDeviceProp device{};
    if (!check_cuda(cudaGetDeviceProperties(&device, 0),
                    "cudaGetDeviceProperties")) {
        return 1;
    }
    std::cerr << "GPU: " << device.name << '\n';
    std::cerr << "Compute capability: " << device.major << '.'
              << device.minor << '\n';

    const bool faulty = (implementation == "faulty_tile");
    std::vector<float> actual;
    if (!run_matmul(a, b, n, faulty, device, actual)) {
        return 1;
    }

    std::cout << std::setprecision(9) << "{\"n\":" << n << ",\"actual\":[";
    for (int i = 0; i < n * n; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << actual[i];
    }
    std::cout << "]}" << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 ||
        (std::string(argv[1]) != "correct" &&
         std::string(argv[1]) != "faulty" &&
         std::string(argv[1]) != "replay")) {
        std::cerr << "Usage: ./matmul_proof correct|faulty|replay\n";
        return 2;
    }

    const std::string mode = argv[1];
    if (mode == "replay") {
        return run_replay();
    }
    return run_campaign(mode == "faulty");
}
