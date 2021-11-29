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

static void fill_vectors(float *a, float *b, size_t size) {
    for (size_t i = 1; i <= size; i++) {
        a[i] = i * 0.1;
        b[i] = (size - i) * 0.1;
    }
}

static void BM_dot_product_avx2(benchmark::State& state) {
    size_t size = 128;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2(a, b, size);
    }
}

static void BM_dot_product_avx2_two_accumulators(benchmark::State& state) {
    size_t size = 128;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2_two_accumulators(a, b, size);
    }
}

static void BM_dot_product(benchmark::State& state) {
    size_t size = 128;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product(a, b, size);
    }
}

// BENCHMARK(BM_add_float32_avx2);
// BENCHMARK(BM_add_float32_SIMD);
// BENCHMARK(BM_add_float32);
// BENCHMARK(BM_mul_float32_avx2);
// BENCHMARK(BM_mul_float32_SIMD);
// BENCHMARK(BM_mul_float32);
BENCHMARK(BM_dot_product_avx2);
BENCHMARK(BM_dot_product_avx2_two_accumulators);
BENCHMARK(BM_dot_product);
