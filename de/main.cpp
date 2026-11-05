#include "kernel.h"
#include <vector>
#include <cstring>
#include <iomanip>

static void generate_population(float *population, int pop, int dim,
                                float xmin, float xmax, unsigned long seed) {
    srand((unsigned)seed);
    for (int i = 0; i < pop * dim; ++i)
        population[i] = getRandom(xmin, xmax);
}

static void summarize(const char *mode, int func, int dim, int pop,
                      const std::vector<double> &fits,
                      const std::vector<double> &times) {
    const int runs = (int)fits.size();
    double mf = 0.0, mt = 0.0;
    for (int r = 0; r < runs; ++r) { mf += fits[r]; mt += times[r]; }
    mf /= runs; mt /= runs;

    double sf = 0.0, st = 0.0;
    for (int r = 0; r < runs; ++r) {
        sf += (fits[r] - mf) * (fits[r] - mf);
        st += (times[r] - mt) * (times[r] - mt);
    }
    if (runs > 1) {
        sf = sqrt(sf / (runs - 1));
        st = sqrt(st / (runs - 1));
    }

    printf("# %s F%d D=%d NP=%d | error mean=%.6e std=%.6e | time mean=%.3f ms std=%.3f\n",
           mode, func, dim, pop, mf, sf, mt, st);
}

int main(int argc, char **argv) {
    int dim = 10, pop = 50, func = 3, runs = 10;
    std::string mode = "both";
    unsigned long baseSeed = 123456789UL;

    if (argc >= 4) {
        dim = std::stoi(argv[1]);
        pop = std::stoi(argv[2]);
        func = std::stoi(argv[3]);
    }
    if (argc >= 5) runs = std::stoi(argv[4]);
    if (argc >= 6) mode = argv[5];
    if (argc >= 7) baseSeed = std::stoul(argv[6]);

    if (dim < 2 || pop < 4 || func < 1 || func > 4 || runs < 1 ||
        (mode != "cpu" && mode != "gpu" && mode != "both")) {
        printf("Usage: %s <dim> <pop> <func 1..4> [runs] [cpu|gpu|both] [base_seed]\n", argv[0]);
        printf("  func: 1=Sphere 2=Rosenbrock 3=Griewank 4=Rastrigin\n");
        return 1;
    }

    float xmin, xmax;
    get_bounds(func, xmin, xmax);

    const size_t count = (size_t)pop * dim;
    float *initial = new float[count];
    float *cpuPop  = new float[count];
    float *gpuPop  = new float[count];
    float *cpuBest = new float[dim];
    float *gpuBest = new float[dim];

    std::vector<double> cpuFits, cpuTimes, gpuFits, gpuTimes, speedups;
    cpuFits.reserve(runs); cpuTimes.reserve(runs);
    gpuFits.reserve(runs); gpuTimes.reserve(runs); speedups.reserve(runs);

    // One warm-up outside measured runs so CUDA context creation does not pollute run 1.
    if (mode == "gpu" || mode == "both") {
        generate_population(gpuPop, pop, dim, xmin, xmax, baseSeed ^ 0xA5A5A5A5UL);
        float warm = 0.0f;
        cuda_de(gpuPop, gpuBest, pop, dim, func, baseSeed, &warm);
    }

    printf("func,dim,pop,run,error,time_ms,mode,seed\n");

    for (int r = 0; r < runs; ++r) {
        const unsigned long runSeed = baseSeed + (unsigned long)r;
        const unsigned long initSeed = baseSeed + 1000003UL + (unsigned long)r;

        // Exactly the same starting population for CPU and GPU in this run.
        generate_population(initial, pop, dim, xmin, xmax, initSeed);

        if (mode == "cpu" || mode == "both") {
            memcpy(cpuPop, initial, count * sizeof(float));
            float ms = 0.0f;
            sequential_de(cpuPop, cpuBest, pop, dim, func, runSeed, &ms);
            const double fit = host_fitness_function(cpuBest, dim, func);
            cpuFits.push_back(fit); cpuTimes.push_back(ms);
            printf("%d,%d,%d,%d,%.6e,%.3f,cpu,%lu\n",
                   func, dim, pop, r + 1, fit, ms, runSeed);
        }

        if (mode == "gpu" || mode == "both") {
            memcpy(gpuPop, initial, count * sizeof(float));
            float ms = 0.0f;
            cuda_de(gpuPop, gpuBest, pop, dim, func, runSeed, &ms);
            const double fit = host_fitness_function(gpuBest, dim, func);
            gpuFits.push_back(fit); gpuTimes.push_back(ms);
            printf("%d,%d,%d,%d,%.6e,%.3f,gpu,%lu\n",
                   func, dim, pop, r + 1, fit, ms, runSeed);
        }

        if (mode == "both") {
            const double s = gpuTimes.back() > 0.0 ? cpuTimes.back() / gpuTimes.back() : 0.0;
            speedups.push_back(s);
            printf("# pair run=%d speedup_cpu_over_gpu=%.4f\n", r + 1, s);
        }
    }

    if (mode == "cpu" || mode == "both")
        summarize("cpu", func, dim, pop, cpuFits, cpuTimes);
    if (mode == "gpu" || mode == "both")
        summarize("gpu", func, dim, pop, gpuFits, gpuTimes);

    if (mode == "both") {
        double meanSpeedup = 0.0;
        for (double s : speedups) meanSpeedup += s;
        meanSpeedup /= speedups.size();
        const double ratioOfMeans =
            (cpuTimes.empty() || gpuTimes.empty()) ? 0.0 :
            ([&]() {
                double c = 0.0, g = 0.0;
                for (double v : cpuTimes) c += v;
                for (double v : gpuTimes) g += v;
                return (g > 0.0) ? (c / cpuTimes.size()) / (g / gpuTimes.size()) : 0.0;
            })();
        printf("# speedup paired_mean=%.4f ratio_of_mean_times=%.4f\n",
               meanSpeedup, ratioOfMeans);
    }

    delete[] initial;
    delete[] cpuPop;
    delete[] gpuPop;
    delete[] cpuBest;
    delete[] gpuBest;
    return 0;
}
