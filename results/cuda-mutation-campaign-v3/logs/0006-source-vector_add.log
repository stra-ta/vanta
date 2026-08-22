// Vanta vector-add proof: second kernel, validating elementwise y = a*x + y
// against a CPU reference, with a deliberately broken tail variant.
//
// Modes:
//   vector_add_proof correct
//   vector_add_proof faulty
//   vector_add_proof replay      (reads a vector-add failure case from stdin)
//
// Replay line protocol:
//   <implementation>
//   <n>
//   <block_threads>
//   <a>
//   <x0> <x1> ... <x_{n-1}>
//   <y0> <y1> ... <y_{n-1}>

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

__global__ void vector_add_kernel(float a, const float* x, const float* y,
                                  float* out, int n, bool faulty) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float value = a * x[i] + y[i];
    // The broken tail variant drops the final element (writes 0).
    if (!faulty || i < n - 1) {
        out[i] = value;
    } else {
        out[i] = 0.0f;
    }
}

std::vector<float> make_vector(int n, int pattern) {
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i) {
        if (pattern == 0) {
            values[i] = 1.0f;
        } else if (pattern == 1) {
            values[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        } else {
            // Anchored at +1 so the tail index is never zero; this keeps the
            // faulty_tail defect observable at every tested size (the old
            // (i%7)-3 form produced a zero at N=32 and N=256).
            values[i] = static_cast<float>((i % 7) + 1);
        }
    }
    return values;
}

void reference_vadd(float a, const std::vector<float>& x,
                    const std::vector<float>& y, std::vector<float>& out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = a * x[i] + y[i];
    }
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

// Run one vector-add case, returning the full host output vector.
bool run_case(float a, const std::vector<float>& x,
              const std::vector<float>& y, int block_size, bool faulty,
              std::vector<float>& out_host) {
    const int n = static_cast<int>(x.size());
    if (n <= 0 || block_size <= 0) {
        return false;
    }
    const int grid = (n + block_size - 1) / block_size;

    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_out = nullptr;
    bool ok = true;

    if (!check_cuda(cudaMalloc(&device_x, n * sizeof(float)),
                    "cudaMalloc(x)")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&device_y, n * sizeof(float)),
                    "cudaMalloc(y)")) {
        cudaFree(device_x);
        return false;
    }
    if (!check_cuda(cudaMalloc(&device_out, n * sizeof(float)),
                    "cudaMalloc(out)")) {
        cudaFree(device_x);
        cudaFree(device_y);
        return false;
    }

    if (!check_cuda(
            cudaMemcpy(device_x, x.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(x)")) {
        ok = false;
        goto cleanup;
    }
    if (!check_cuda(
            cudaMemcpy(device_y, y.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(y)")) {
        ok = false;
        goto cleanup;
    }

    vector_add_kernel<<<grid, block_size>>>(a, device_x, device_y, device_out,
                                            n, faulty);
    if (!check_cuda(cudaDeviceSynchronize(), "vector_add_kernel")) {
        ok = false;
        goto cleanup;
    }

    out_host.resize(n);
    if (!check_cuda(
            cudaMemcpy(out_host.data(), device_out, n * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(out)")) {
        ok = false;
        goto cleanup;
    }

cleanup:
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_out);
    return ok;
}

void write_failure(float a, const std::vector<float>& x,
                   const std::vector<float>& y, int pattern, int block_size,
                   const std::vector<float>& expected,
                   const std::vector<float>& actual_vec,
                   const cudaDeviceProp& device) {
    const int n = static_cast<int>(x.size());
    const int idx = n - 1;  // tail defect records the last element
    std::ofstream file("failure.json");
    file << std::setprecision(9);
    file << "{\n";
    file << "  \"schema\": \"vanta.failure.v0\",\n";
    file << "  \"kernel\": \"vector_add\",\n";
    file << "  \"implementation\": \"faulty_tail\",\n";
    file << "  \"input\": {\n";
    file << "    \"dtype\": \"float32\",\n";
    file << "    \"shape\": [" << n << "],\n";
    file << "    \"values\": [";
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            file << ", ";
        }
        file << x[i];
    }
    file << "]\n";
    file << "  },\n";
    file << "  \"params\": {\"a\": " << a << "},\n";
    file << "  \"aux\": {\"y\": [";
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            file << ", ";
        }
        file << y[i];
    }
    file << "]},\n";
    file << "  \"launch\": {\n";
    file << "    \"block_threads\": " << block_size << ",\n";
    file << "    \"grid_blocks\": 1\n";
    file << "  },\n";
    file << "  \"environment\": {\n";
    file << "    \"gpu\": \"" << device.name << "\",\n";
    file << "    \"compute_capability\": \"" << device.major << "."
         << device.minor << "\"\n";
    file << "  },\n";
    file << "  \"observed\": {\n";
    file << "    \"kind\": \"correctness_mismatch\",\n";
    file << "    \"index\": " << idx << ",\n";
    file << "    \"pattern\": " << pattern << ",\n";
    file << "    \"expected\": " << expected[idx] << ",\n";
    file << "    \"actual\": " << actual_vec[idx] << "\n";
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
    std::cout << "Compute capability: " << device.major << '.' << device.minor
              << '\n';

    const float a = 2.0f;
    const std::vector<int> sizes = {31, 32, 33, 255, 256, 257};
    const std::vector<int> block_sizes = {32, 64, 128, 256};
    const std::vector<int> patterns = {0, 1, 2};

    int pass_count = 0;
    int mismatch_count = 0;

    for (int n : sizes) {
        for (int pattern : patterns) {
            std::vector<float> x = make_vector(n, pattern);
            std::vector<float> y = make_vector(n, pattern);
            std::vector<float> expected;
            reference_vadd(a, x, y, expected);

            for (int block_size : block_sizes) {
                std::vector<float> actual;
                if (!run_case(a, x, y, block_size, faulty, actual)) {
                    return 1;
                }

                bool passed = true;
                int first_bad = -1;
                for (int i = 0; i < n; ++i) {
                    const double tol =
                        1e-5 * std::max(1.0, std::abs(static_cast<double>(expected[i])));
                    if (std::abs(actual[i] - expected[i]) > tol) {
                        passed = false;
                        first_bad = i;
                        break;
                    }
                }

                if (passed) {
                    ++pass_count;
                    std::cout << "PASS n=" << n << " pattern=" << pattern
                              << " block=" << block_size << '\n';
                } else {
                    ++mismatch_count;
                    std::cout << "MISMATCH n=" << n << " pattern=" << pattern
                              << " block=" << block_size << " index="
                              << first_bad << " expected=" << expected[first_bad]
                              << " actual=" << actual[first_bad] << '\n';
                    if (faulty && mismatch_count == 1) {
                        write_failure(a, x, y, pattern, block_size, expected,
                                      actual, device);
                    }
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
    int block_size = 0;
    float a = 0.0f;

    if (!(std::cin >> implementation >> n >> block_size >> a)) {
        std::cerr << "Invalid replay header\n";
        return 2;
    }
    if (implementation != "faulty_tail") {
        std::cerr << "Unsupported implementation: " << implementation << '\n';
        return 2;
    }
    if (n <= 0 || block_size <= 0) {
        std::cerr << "Invalid shape or block size\n";
        return 2;
    }

    std::vector<float> x(n), y(n);
    for (float& value : x) {
        if (!(std::cin >> value)) {
            std::cerr << "Not enough x values\n";
            return 2;
        }
    }
    for (float& value : y) {
        if (!(std::cin >> value)) {
            std::cerr << "Not enough y values\n";
            return 2;
        }
    }

    cudaDeviceProp device{};
    if (!check_cuda(cudaGetDeviceProperties(&device, 0),
                    "cudaGetDeviceProperties")) {
        return 1;
    }
    std::cerr << "GPU: " << device.name << '\n';
    std::cerr << "Compute capability: " << device.major << '.' << device.minor
              << '\n';

    std::vector<float> actual;
    if (!run_case(a, x, y, block_size, /*faulty=*/true, actual)) {
        return 1;
    }

    std::vector<float> expected;
    reference_vadd(a, x, y, expected);
    const int idx = n - 1;

    std::cout << std::setprecision(9) << "{\"n\":" << n
              << ",\"block_threads\":" << block_size
              << ",\"expected\":" << expected[idx]
              << ",\"actual\":" << actual[idx] << "}" << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 ||
        (std::string(argv[1]) != "correct" &&
         std::string(argv[1]) != "faulty" &&
         std::string(argv[1]) != "replay")) {
        std::cerr << "Usage: ./vector_add_proof correct|faulty|replay\n";
        return 2;
    }

    const std::string mode = argv[1];
    if (mode == "replay") {
        return run_replay();
    }
    return run_campaign(mode == "faulty");
}
