# include "dotproduct.h"

float dot_product_avx2(float *a, float *b, size_t size) {
    __m256 c = _mm256_setzero_ps();
     size_t limit = size - 7;

     for (size_t i = 0; i < limit; i += 8) {
      __m256 buf1 = _mm256_loadu_ps(a + i);
      __m256 buf2 = _mm256_loadu_ps(b + i);
      c = _mm256_fmadd_ps(buf1, buf2, c);
    }

    float buffer[8] = {};
    _mm256_storeu_ps(buffer, c);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

float dot_product_avx2_two_accumulators(float *a, float *b, size_t size) {
  int accumulators = 2;
  int step = accumulators * 8;
  const float* limit = a + size;

  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();

  {

    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);

    c0 = _mm256_mul_ps(buf1, buf2);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);

    c1 = _mm256_mul_ps(buf1, buf2);

    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_fmadd_ps(buf1, buf2, c0);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);
    c1 = _mm256_fmadd_ps(buf1, buf2, c1);
  }

  c0 = _mm256_add_ps(c0, c1);
  
  float buffer[8] = {};
  _mm256_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

float dot_product(float *a, float *b, size_t size) {
  float c = 0;
  for (size_t i = 0; i < size; i++) {
    c += a[i] * b[i];
  }
  return c;
}