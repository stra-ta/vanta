// Vanta reduction proof: single-block sum reduction validation.
//
// This file is intentionally one self-contained program. It is the first
// Vanta artifact, not the final framework. It supports three modes:
//
//   reduction_proof correct
//       Run the full structured boundary campaign with the correct kernel.
//
//   reduction_proof faulty
//       Run the same campaign with the deliberately broken tail kernel.
//
//   reduction_proof replay
//       Read one failure case from stdin and run it once. The JSON
//       articulation is owned by Python; this program only consumes a
//       small line protocol:
//
//           <implementation>
//           <n>
//           <block_threads>
//           <value> <value> ... <value>
//
// Correctness evidence in this milestone comes from a straightforward
// CPU double-precision sum. The comparison tolerance is documented in
// the README and must not be silently loosened.

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

// Single-block reduction with a configurable defect.
//
// The faulty tail variant drops the final element. It is not a memory
// bug; it is a logical error, which is why Compute Sanitizer may stay
// clean while differential validation still catches it.
__global__ void reduce_sum_kernel(const float* input, float* output, int n,
                                   bool faulty) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        if (!faulty || i < n - 1) {
            local_sum += input[i];
        }
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = shared[0];
    }
}

std::vector<float> make_input(int n, int pattern) {
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i) {
        if (pattern == 0) {
            values[i] = 1.0f;
        } else if (pattern == 1) {
            values[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        } else {
            values[i] = static_cast<float>((i % 7) - 3);
        }
    }
    return values;
}

double reference_sum(const std::vector<float>& values) {
    double result = 0.0;
    for (float value : values) {
        result += static_cast<double>(value);
    }
    return result;
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

// The tolerance is absolute-relative: scale with the magnitude of the
// expected sum, but never below a small floor. For sums of small integer
// values this is effectively an exact comparison.
double tolerance_for(double expected) {
    return 1e-5 * std::max(1.0, std::abs(expected));
}

// Run one case. Returns true when the actual result matches the reference.
bool run_case(const std::vector<float>& input, int block_size, bool faulty,
              const cudaDeviceProp& device, double& actual_out) {
    const int n = static_cast<int>(input.size());
    float* device_input = nullptr;
    float* device_output = nullptr;

    if (!check_cuda(
            cudaMalloc(&device_input, input.size() * sizeof(float)),
            "cudaMalloc(input)")) {
        return false;
    }
    if (!check_cuda(cudaMalloc(&device_output, sizeof(float)),
                    "cudaMalloc(output)")) {
        cudaFree(device_input);
        return false;
    }
    if (!check_cuda(
            cudaMemcpy(device_input, input.data(),
                       input.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy(input)")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }

    reduce_sum_kernel<<<1, block_size, block_size * sizeof(float)>>>(
        device_input, device_output, n, faulty);

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }

    float actual = 0.0f;
    if (!check_cuda(
            cudaMemcpy(&actual, device_output, sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(output)")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }

    cudaFree(device_input);
    cudaFree(device_output);

    actual_out = static_cast<double>(actual);
    (void)device;
    return true;
}

void write_failure(const std::vector<float>& input, int pattern, int block_size,
                   double expected, float actual,
                   const cudaDeviceProp& device) {
    std::ofstream file("failure.json");
    file << std::setprecision(9);
    file << "{\n";
    file << "  \"schema\": \"vanta.failure.v0\",\n";
    file << "  \"kernel\": \"reduce_sum\",\n";
    file << "  \"implementation\": \"faulty_tail\",\n";
    file << "  \"input\": {\n";
    file << "    \"dtype\": \"float32\",\n";
    file << "    \"shape\": [" << input.size() << "],\n";
    file << "    \"values\": [";
    for (size_t i = 0; i < input.size(); ++i) {
        if (i > 0) {
            file << ", ";
        }
        file << input[i];
    }
    file << "]\n";
    file << "  },\n";
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
    file << "    \"pattern\": " << pattern << ",\n";
    file << "    \"expected\": " << expected << ",\n";
    file << "    \"actual\": " << actual << "\n";
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

    const std::vector<int> sizes = {31, 32, 33, 255, 256, 257};
    const std::vector<int> block_sizes = {32, 64, 128, 256};
    const std::vector<int> patterns = {0, 1, 2};

    int pass_count = 0;
    int mismatch_count = 0;

    for (int n : sizes) {
        for (int pattern : patterns) {
            std::vector<float> input = make_input(n, pattern);
            double expected = reference_sum(input);

            for (int block_size : block_sizes) {
                double actual = 0.0;
                if (!run_case(input, block_size, faulty, device, actual)) {
                    return 1;
                }

                const bool passed =
                    std::abs(actual - expected) <= tolerance_for(expected);

                if (passed) {
                    ++pass_count;
                    std::cout << "PASS n=" << n << " pattern=" << pattern
                              << " block=" << block_size << '\n';
                } else {
                    ++mismatch_count;
                    std::cout << "MISMATCH n=" << n << " pattern=" << pattern
                              << " block=" << block_size << " expected="
                              << expected << " actual=" << actual << '\n';
                    if (faulty && mismatch_count == 1) {
                        write_failure(input, pattern, block_size, expected,
                                      static_cast<float>(actual), device);
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

    if (!(std::cin >> implementation >> n >> block_size)) {
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

    std::vector<float> input(n);
    for (float& value : input) {
        if (!(std::cin >> value)) {
            std::cerr << "Not enough input values\n";
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

    double actual = 0.0;
    if (!run_case(input, block_size, /*faulty=*/true, device, actual)) {
        return 1;
    }

    std::cout << std::setprecision(9) << "{\"n\":" << n
              << ",\"block_threads\":" << block_size
              << ",\"expected\":" << reference_sum(input)
              << ",\"actual\":" << actual << "}" << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 ||
        (std::string(argv[1]) != "correct" &&
         std::string(argv[1]) != "faulty" &&
         std::string(argv[1]) != "replay")) {
        std::cerr << "Usage: ./reduction_proof correct|faulty|replay\n";
        return 2;
    }

    const std::string mode = argv[1];
    if (mode == "replay") {
        return run_replay();
    }
    return run_campaign(mode == "faulty");
}
