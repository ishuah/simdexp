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

static void BM_dot_product_avx2(benchmark::State& state) {
    float a[16] = {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7};
    float b[16] = {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8};
    float c;
    size_t size = 16;

    for (auto _ : state) {
        c = dot_product_avx2(a, b, size);
    }
}

static void BM_dot_product(benchmark::State& state) {
    float a[16] = {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7};
    float b[16] = {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8};
    float c;
    size_t size = 16;

    for (auto _ : state) {
        c = dot_product(a, b, size);
    }
}

BENCHMARK(BM_add_float32_avx2);
BENCHMARK(BM_add_float32_SIMD);
BENCHMARK(BM_add_float32);
BENCHMARK(BM_mul_float32_avx2);
BENCHMARK(BM_mul_float32_SIMD);
BENCHMARK(BM_mul_float32);
BENCHMARK(BM_dot_product_avx2);
BENCHMARK(BM_dot_product);
