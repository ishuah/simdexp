#include <addition.h>
#include <multiplication.h>
#include <dotproduct.h>
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

static void BM_mul_float32_avx2(benchmark::State& state) {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    for (auto _ : state) {
        mul_float32_avx2(a, b, c);
    }
}

static void BM_mul_float32_SIMD(benchmark::State& state) {
    float a[4] = {0.1, 0.2, 0.3, 0.4};
    float b[4] = {0.5, 0.6, 0.7, 0.8};
    float c[4] = {};
    for (auto _ : state) {
        mul_float32_SIMD(a, b, c);
    }
}

static void BM_mul_float32(benchmark::State& state) {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    for (auto _ : state) {
        mul_float32(a, b, c);
    }
}

static void BM_dot_product_avx2_fma(benchmark::State& state) {
    size_t size = state.range(0);
    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2(a, b, size);
    }
    state.counters["input_size"] = (size * 2 * sizeof(float)) / (1024.0* 1024.0);
}

static void BM_dot_product_avx2_two_accumulators(benchmark::State& state) {
    size_t size = state.range(0);
    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2_two_accumulators(a, b, size);
    }
    state.counters["input_size"] = (size * 2 * sizeof(float)) / (1024.0* 1024.0);
}

static void BM_dot_product_avx2_four_accumulators(benchmark::State& state) {
    size_t size = state.range(0);
    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2_four_accumulators(a, b, size);
    }
    state.counters["input_size"] = (size * 2 * sizeof(float)) / (1024.0* 1024.0);
}

static void BM_dot_product_avx2_multiple_accumulators(benchmark::State& state) {
    size_t size = state.range(1);
    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2_multiple_accumulators(a, b, size, state.range(0));
    }
    state.counters["input_size"] = (size * 2 * sizeof(float)) / (1024.0* 1024.0);
}

static void BM_dot_product_naive(benchmark::State& state) {
    size_t size = state.range(0);
    float *a = (float *)malloc(sizeof(float) * size);
    float *b = (float *)malloc(sizeof(float) * size);
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product(a, b, size);
    }
    state.counters["input_size"] = (size * 2 * sizeof(float)) / (1024.0* 1024.0);
}

// BENCHMARK(BM_add_float32_avx2);
// BENCHMARK(BM_add_float32_SIMD);
// BENCHMARK(BM_add_float32);
// BENCHMARK(BM_mul_float32_avx2);
// BENCHMARK(BM_mul_float32_SIMD);
// BENCHMARK(BM_mul_float32);

BENCHMARK(BM_dot_product_avx2_fma)
    ->Args({131072})
    ->Args({262144})
    ->Args({524288})
    ->Args({1048576})
    ->Args({2097152})
    ->Args({4194304})
    ->Args({8388608})
    ->Args({16777216})
    ->Args({33554432});
BENCHMARK(BM_dot_product_avx2_two_accumulators)
    ->Args({131072})
    ->Args({262144})
    ->Args({524288})
    ->Args({1048576})
    ->Args({2097152})
    ->Args({4194304})
    ->Args({8388608})
    ->Args({16777216})
    ->Args({33554432});
BENCHMARK(BM_dot_product_avx2_four_accumulators)
    ->Args({131072})
    ->Args({262144})
    ->Args({524288})
    ->Args({1048576})
    ->Args({2097152})
    ->Args({4194304})
    ->Args({8388608})
    ->Args({16777216})
    ->Args({33554432});
BENCHMARK(BM_dot_product_avx2_multiple_accumulators)
    ->Args({2,131072})
    ->Args({2,262144})
    ->Args({2,524288})
    ->Args({2,1048576})
    ->Args({2,2097152})
    ->Args({2,4194304})
    ->Args({2,8388608})
    ->Args({2,16777216})
    ->Args({2,33554432});
BENCHMARK(BM_dot_product_avx2_multiple_accumulators)
    ->Args({4,131072})
    ->Args({4,262144})
    ->Args({4,524288})
    ->Args({4,1048576})
    ->Args({4,2097152})
    ->Args({4,4194304})
    ->Args({4,8388608})
    ->Args({4,16777216})
    ->Args({4,33554432});
BENCHMARK(BM_dot_product_naive)
    ->Args({131072})
    ->Args({262144})
    ->Args({524288})
    ->Args({1048576})
    ->Args({2097152})
    ->Args({4194304})
    ->Args({8388608})
    ->Args({16777216})
    ->Args({33554432});
    