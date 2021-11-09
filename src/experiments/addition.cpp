#include "addition.h"

void add_float32_simd(float buf1[8], float buf2[8], float res[8]) {
    __m256 a = _mm256_load_ps(buf1);
    __m256 b = _mm256_load_ps(buf2);
    __m256 c = _mm256_add_ps(a, b);
    _mm256_storeu_ps(res, c);
}

void add_float32(float a[8], float b[8], float result[8]) {
    for (int i = 0; i < sizeof(a); i++){
        result[i] = a[i] + b[i];
    }
}
