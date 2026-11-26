// Counter-access probe only; not a representative performance benchmark.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void counter_probe(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = float(i);
    for (int k = 0; k < 1024; ++k) x = fmaf(x, 0.999f, 0.001f);
    data[i] = x;
}

int main() {
    float* data = nullptr;
    cudaError_t status = cudaMalloc(&data, 1024 * 256 * sizeof(float));
    if (status == cudaSuccess) {
        counter_probe<<<1024, 256>>>(data);
        status = cudaDeviceSynchronize();
    }
    std::printf("CUDA status: %s\n", cudaGetErrorString(status));
    if (data) cudaFree(data);
    return status == cudaSuccess ? 0 : 1;
}
