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

static void BM_lemires_dot_product_256(benchmark::State& state) {
    size_t size = state.range(0);
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = lemires_dot_product_avx2(a, b, size);
    }
}

static void BM_dot_product_avx2(benchmark::State& state) {
    size_t size = state.range(0);
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2(a, b, size);
    }
}

static void BM_dot_product_sse(benchmark::State& state) {
    size_t size = state.range(0);
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_sse(a, b, size);
    }
}

static void BM_dot_product_avx2_multiple_accumulators(benchmark::State& state) {
    size_t size = state.range(1);
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_avx2_multiple_accumulators(a, b, size, state.range(0));
    }
}

static void BM_dot_product_sse_multiple_accumulators(benchmark::State& state) {
    size_t size = state.range(1);
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    for (auto _ : state) {
        c = dot_product_sse_multiple_accumulators(a, b, size, state.range(0));
    }
}

static void BM_dot_product(benchmark::State& state) {
    size_t size = state.range(0);
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

// BENCHMARK(BM_lemires_dot_product_256)
//     ->Args({128});
    // ->Args({1024})
    // ->Args({4096})
    // ->Args({8192})
    // ->Args({262144})
    // ->Args({524288});
BENCHMARK(BM_dot_product_avx2)
    ->Args({128});
    // ->Args({1024})
    // ->Args({4096})
    // ->Args({8192})
    // ->Args({262144})
    // ->Args({524288});
    
// BENCHMARK(BM_dot_product_sse)
//     ->Args({128})
//     ->Args({1024})
//     ->Args({4096})
//     ->Args({8192})
//     ->Args({262144})
//     ->Args({524288});
BENCHMARK(BM_dot_product_avx2_multiple_accumulators)
    ->Args({2,128});
    // ->Args({2,1024})
    // ->Args({2,4096})
    // ->Args({2,8192})
    // ->Args({2,262144})
    // ->Args({2,524288});
// BENCHMARK(BM_dot_product_avx2_multiple_accumulators)
//     ->Args({4,128});
    // ->Args({4,1024})
    // ->Args({4,4096})
    // ->Args({4,8192})
    // ->Args({4,262144})
    // ->Args({4,524288});
// BENCHMARK(BM_dot_product_sse_multiple_accumulators)
//     ->Args({2,128})
//     ->Args({2,1024})
//     ->Args({2,4096})
//     ->Args({2,8192})
//     ->Args({2,262144})
//     ->Args({2,524288});
// BENCHMARK(BM_dot_product_sse_multiple_accumulators)
//     ->Args({3,128})
//     ->Args({3,1024})
//     ->Args({3,4096})
//     ->Args({3,8192})
//     ->Args({3,262144})
//     ->Args({3,524288});
// BENCHMARK(BM_dot_product_sse_multiple_accumulators)
//     ->Args({4,128})
//     ->Args({4,1024})
//     ->Args({4,4096})
//     ->Args({4,8192})
//     ->Args({4,262144})
//     ->Args({4,524288});
BENCHMARK(BM_dot_product)
    ->Args({128});
    // ->Args({1024})
    // ->Args({4096})
    // ->Args({8192})
    // ->Args({262144})
    // ->Args({524288});