#include "kernel.h"

void get_bounds(int func, float &xmin, float &xmax) {
    switch (func) {
        case 1: case 2: xmin = -100.f; xmax = 100.f; break;
        case 3:         xmin = -600.f; xmax = 600.f; break;
        case 4:         xmin = -5.f;   xmax = 5.f;   break;
        default:        xmin = -100.f; xmax = 100.f; break;
    }
}

// We report EFV = f(x) - f*, therefore the optimum is 0 for all functions.
float host_fitness_function(const float *x, int D, int func) {
    float res = 0.0f, sum = 0.0f, product = 1.0f;
    const float PI2 = 6.2831853071795864769f;

    switch (func) {
        case 1: // Shifted Sphere with o=0, bias removed => EFV
            for (int i = 0; i < D; ++i) res += x[i] * x[i];
            break;

        case 2: // Shifted Rosenbrock, z = x - o + 1, here o=0
            for (int i = 0; i < D - 1; ++i) {
                const float zi = x[i] + 1.0f;
                const float zip1 = x[i + 1] + 1.0f;
                const float t = zi * zi - zip1;
                res += 100.0f * t * t + (zi - 1.0f) * (zi - 1.0f);
            }
            break;

        case 3: // Shifted Griewank with o=0, bias removed => EFV
            for (int i = 0; i < D; ++i) {
                sum += x[i] * x[i] / 4000.0f;
                product *= cosf(x[i] / sqrtf((float)(i + 1)));
            }
            res = sum - product + 1.0f;
            break;

        case 4: // Shifted Rastrigin with o=0, bias removed => EFV
            for (int i = 0; i < D; ++i)
                res += x[i] * x[i] - 10.0f * cosf(PI2 * x[i]) + 10.0f;
            break;

        default:
            return INFINITY;
    }
    return res;
}

float getRandom(float low, float high) {
    return low + (high - low) * (float)(rand() / (RAND_MAX + 1.0));
}

float getRandomClamped() {
    return (float)rand() / (float)RAND_MAX;
}

void sequential_de(float *population, float *gBest, int N, int D, int func,
                   unsigned long seed, float *elapsed_ms)
{
    const auto t0 = std::chrono::steady_clock::now();
    srand((unsigned)seed);

    float xmin, xmax;
    get_bounds(func, xmin, xmax);

    float *fitness = new float[N];
    float *mutant  = new float[(size_t)N * D];
    float *trial   = new float[(size_t)N * D];

    const long MaxFES = 10000L * D;
    long NFE = 0;

    // Initial population fitness: N evaluations.
    for (int i = 0; i < N && NFE < MaxFES; ++i) {
        fitness[i] = host_fitness_function(&population[i * D], D, func);
        ++NFE;
    }

    while (NFE < MaxFES) {
        // 1) Mutation DE/rand/1.
        for (int i = 0; i < N; ++i) {
            int r1, r2, r3;
            do { r1 = rand() % N; } while (r1 == i);
            do { r2 = rand() % N; } while (r2 == i || r2 == r1);
            do { r3 = rand() % N; } while (r3 == i || r3 == r1 || r3 == r2);

            for (int j = 0; j < D; ++j) {
                mutant[i * D + j] = population[r1 * D + j]
                    + F * (population[r2 * D + j] - population[r3 * D + j]);
            }
        }

        // 2) Binomial crossover + forced dimension + clamping.
        for (int i = 0; i < N; ++i) {
            const int k = rand() % D;
            for (int j = 0; j < D; ++j) {
                const float r = getRandomClamped();
                float v = (r <= CR || j == k)
                    ? mutant[i * D + j]
                    : population[i * D + j];

                if (v < xmin) v = xmin;
                if (v > xmax) v = xmax;
                trial[i * D + j] = v;
            }
        }

        // 3) Evaluation + greedy selection.
        // The last generation may be partial so NFE lands exactly on MaxFES.
        for (int i = 0; i < N && NFE < MaxFES; ++i) {
            const float trialFitness = host_fitness_function(&trial[i * D], D, func);
            ++NFE;

            if (trialFitness <= fitness[i]) {
                for (int j = 0; j < D; ++j)
                    population[i * D + j] = trial[i * D + j];
                fitness[i] = trialFitness;
            }
        }
    }

    int best = 0;
    for (int i = 1; i < N; ++i)
        if (fitness[i] < fitness[best]) best = i;

    for (int j = 0; j < D; ++j)
        gBest[j] = population[best * D + j];

    delete[] fitness;
    delete[] mutant;
    delete[] trial;

    const auto t1 = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
}
