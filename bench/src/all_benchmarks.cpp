#include <addition.h>
#include "benchmark/benchmark.h"

using namespace std;

static void BM_add_float32_avx2(benchmark::State& state) {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    for (auto _ : state) {
        add_float32_avx2(a, b, c);
    }
}

static void BM_add_float32_SIMD(benchmark::State& state) {
    float a[4] = {0.1, 0.2, 0.3, 0.4};
    float b[4] = {0.5, 0.6, 0.7, 0.8};
    float c[4] = {};
    for (auto _ : state) {
        add_float32_SIMD(a, b, c);
    }
}

static void BM_add_float32(benchmark::State& state) {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    for (auto _ : state) {
        add_float32(a, b, c);
    }
}

BENCHMARK(BM_add_float32_avx2);
BENCHMARK(BM_add_float32_SIMD);
BENCHMARK(BM_add_float32);
