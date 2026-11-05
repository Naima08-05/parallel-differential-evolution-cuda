#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include "kernel.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return; \
    } \
} while (0)

__device__ float fitness_function(const float *x, int D, int func) {
    float res = 0.0f, sum = 0.0f, product = 1.0f;
    const float PI2 = 6.2831853071795864769f;

    switch (func) {
        case 1:
            for (int i = 0; i < D; ++i) res += x[i] * x[i];
            break;
        case 2:
            for (int i = 0; i < D - 1; ++i) {
                const float zi = x[i] + 1.0f;
                const float zip1 = x[i + 1] + 1.0f;
                const float t = zi * zi - zip1;
                res += 100.0f * t * t + (zi - 1.0f) * (zi - 1.0f);
            }
            break;
        case 3:
            for (int i = 0; i < D; ++i) {
                sum += x[i] * x[i] / 4000.0f;
                product *= cosf(x[i] / sqrtf((float)(i + 1)));
            }
            res = sum - product + 1.0f;
            break;
        case 4:
            for (int i = 0; i < D; ++i)
                res += x[i] * x[i] - 10.0f * cosf(PI2 * x[i]) + 10.0f;
            break;
        default:
            return 1.0e30f;
    }
    return res;
}

__global__ void kernelInitCurand(curandState *states, unsigned long seed, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) curand_init(seed, i, 0, &states[i]);
}

// Initial evaluation is performed on the GPU, not on the CPU.
__global__ void kernelEvaluate(const float *population, float *fitness,
                               int N, int D, int func) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) fitness[i] = fitness_function(&population[i * D], D, func);
}

__global__ void kernelMutation(const float *population, float *mutant,
                               curandState *states, int D, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local = states[i];
    int r1, r2, r3;
    do { r1 = curand(&local) % N; } while (r1 == i);
    do { r2 = curand(&local) % N; } while (r2 == i || r2 == r1);
    do { r3 = curand(&local) % N; } while (r3 == i || r3 == r1 || r3 == r2);

    for (int j = 0; j < D; ++j) {
        mutant[i * D + j] = population[r1 * D + j]
            + F * (population[r2 * D + j] - population[r3 * D + j]);
    }
    states[i] = local;
}

// One thread handles one individual because CURAND state is stored per individual.
__global__ void kernelCrossover(const float *population, const float *mutant,
                                float *trial, curandState *states,
                                int D, int N, float cr,
                                float xmin, float xmax) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local = states[i];
    const int k = curand(&local) % D;

    for (int j = 0; j < D; ++j) {
        const float r = curand_uniform(&local);
        float v = (r <= cr || j == k)
            ? mutant[i * D + j]
            : population[i * D + j];

        if (v < xmin) v = xmin;
        if (v > xmax) v = xmax;
        trial[i * D + j] = v;
    }
    states[i] = local;
}

// activeCount allows an exact MaxFES budget on the last partial generation.
__global__ void kernelSelection(float *population, float *fitness,
                                const float *trial,
                                int activeCount, int D, int func) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= activeCount) return;

    const float fu = fitness_function(&trial[i * D], D, func);
    if (fu <= fitness[i]) {
        for (int j = 0; j < D; ++j)
            population[i * D + j] = trial[i * D + j];
        fitness[i] = fu;
    }
}

extern "C" void cuda_de(float *population, float *gBest, int N, int D,
                        int func, unsigned long seed, float *elapsed_ms) {
    const auto t0 = std::chrono::steady_clock::now();

    float xmin, xmax;
    get_bounds(func, xmin, xmax);

    const size_t popBytes = (size_t)N * D * sizeof(float);
    float *d_pop = nullptr, *d_fit = nullptr, *d_mutant = nullptr, *d_trial = nullptr;
    curandState *d_states = nullptr;
    float *fit = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_pop, popBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_mutant, popBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_trial, popBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_fit, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_states, N * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(d_pop, population, popBytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    kernelInitCurand<<<blocks, threads>>>(d_states, seed, N);
    CUDA_CHECK(cudaPeekAtLastError());

    kernelEvaluate<<<blocks, threads>>>(d_pop, d_fit, N, D, func);
    CUDA_CHECK(cudaPeekAtLastError());

    const long MaxFES = 10000L * D;
    long NFE = N;

    while (NFE < MaxFES) {
        const int activeCount = (int)((MaxFES - NFE) < N ? (MaxFES - NFE) : N);

        kernelMutation<<<blocks, threads>>>(d_pop, d_mutant, d_states, D, N);
        CUDA_CHECK(cudaPeekAtLastError());

        kernelCrossover<<<blocks, threads>>>(d_pop, d_mutant, d_trial, d_states,
                                             D, N, CR, xmin, xmax);
        CUDA_CHECK(cudaPeekAtLastError());

        const int selectionBlocks = (activeCount + threads - 1) / threads;
        kernelSelection<<<selectionBlocks, threads>>>(d_pop, d_fit, d_trial,
                                                       activeCount, D, func);
        CUDA_CHECK(cudaPeekAtLastError());

        NFE += activeCount;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    fit = new float[N];
    CUDA_CHECK(cudaMemcpy(fit, d_fit, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(population, d_pop, popBytes, cudaMemcpyDeviceToHost));

    int best = 0;
    for (int i = 1; i < N; ++i)
        if (fit[i] < fit[best]) best = i;
    for (int j = 0; j < D; ++j)
        gBest[j] = population[best * D + j];

    delete[] fit;
    CUDA_CHECK(cudaFree(d_pop));
    CUDA_CHECK(cudaFree(d_mutant));
    CUDA_CHECK(cudaFree(d_trial));
    CUDA_CHECK(cudaFree(d_fit));
    CUDA_CHECK(cudaFree(d_states));

    const auto t1 = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
}
