#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>
#include <chrono>

// DE/rand/1/bin parameters shared by CPU and GPU.
constexpr float F  = 0.5f;
constexpr float CR = 0.3f;

// Benchmark ids: 1=Sphere, 2=Rosenbrock, 3=Griewank, 4=Rastrigin.
void get_bounds(int func, float &xmin, float &xmax);
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(const float *x, int D, int func);

// Sequential DE baseline (CPU).
// population is modified in place; gBest receives the best final vector.
// elapsed_ms is end-to-end wall time of this call.
void sequential_de(float *population, float *gBest, int N, int D, int func,
                   unsigned long seed, float *elapsed_ms);

// Massively parallel DE (GPU).
// population is modified in place; gBest receives the best final vector.
// elapsed_ms is end-to-end wall time of this call, including allocation,
// H2D/D2H transfers, kernels, synchronization and cleanup.
extern "C" void cuda_de(float *population, float *gBest, int N, int D,
                        int func, unsigned long seed, float *elapsed_ms);

#endif
