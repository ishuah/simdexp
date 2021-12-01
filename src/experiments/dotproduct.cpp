# include "dotproduct.h"
#include <assert.h>
#include <iostream>

void fill_vectors(float *a, float *b, size_t size) {
    for (size_t i = 0; i <= size; i++) {
        a[i] = (i+1) * 0.1;
        b[i] = (size - i) * 0.1;
    }
}

__attribute__((noinline)) float lemires_dot_product_avx2(float *x1, float *x2, size_t len) {
  __m256 sum = _mm256_setzero_ps();
  if (len > 7) {
    size_t limit = len - 7;
    for (size_t i = 0; i < limit; i += 8) {
      __m256 v1 = _mm256_loadu_ps(x1 + i);
      __m256 v2 = _mm256_loadu_ps(x2 + i);
      sum = _mm256_fmadd_ps(v1, v2, sum);
    }
  }
  float buffer[8];
  _mm256_storeu_ps(buffer, sum);
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

__attribute__((noinline)) float konstantins_dot_product_avx2( const float* p1, const float* p2, size_t count )
{
	__m256 acc = _mm256_setzero_ps();
	const float* const p1End = p1 + count;
	for( ; p1 < p1End; p1 += 8, p2 += 8 )
	{
		// Load 2 vectors, 8 floats / each
		const __m256 a = _mm256_loadu_ps( p1 );
		const __m256 b = _mm256_loadu_ps( p2 );
		// vdpps AVX instruction does not compute dot product of 8-wide vectors.
		// Instead, that instruction computes 2 independent dot products of 4-wide vectors.
		const __m256 dp = _mm256_dp_ps( a, b, 0xFF );
		acc = _mm256_add_ps( acc, dp );
	}

	// Add the 2 results into a single float.
	const __m128 low = _mm256_castps256_ps128( acc );	//< Compiles into no instructions. The low half of a YMM register is directly accessible as an XMM register with the same number.
	const __m128 high = _mm256_extractf128_ps( acc, 1 );	//< This one however does need to move data, from high half of a register into low half. vextractf128 instruction does that.
	const __m128 result = _mm_add_ss( low, high );
	return _mm_cvtss_f32( result );
}

__attribute__((noinline)) float dot_product_avx2(float *a, float *b, size_t size) {
    __m256 c = _mm256_setzero_ps();
     size_t limit = size - 7;

     for (size_t i = 0; i < limit; i += 8) {
      __m256 buf1 = _mm256_loadu_ps(a + i);
      __m256 buf2 = _mm256_loadu_ps(b + i);
      c = _mm256_fmadd_ps(buf1, buf2, c);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, c);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

__attribute__((noinline)) float dot_product_sse(float *a, float *b, size_t size) {
    __m128 c = _mm_setzero_ps();
     size_t limit = size - 3;

     for (size_t i = 0; i < limit; i += 4) {
      __m128 buf1 = _mm_loadu_ps(a + i);
      __m128 buf2 = _mm_loadu_ps(b + i);
      c = _mm_fmadd_ps(buf1, buf2, c);
    }

    float buffer[4];
    _mm_storeu_ps(buffer, c);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}

__attribute__((noinline)) float dot_product_avx2_multiple_accumulators(float *a, float *b, size_t size, int accumulators) {
  int step = accumulators * 8;
  const float* limit = a + size;
  assert( 0 == size % accumulators );

  __m256 c0,c1,c2,c3;
  {

    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_mul_ps(buf1, buf2);

    if (accumulators > 1) {
      buf1 = _mm256_loadu_ps(a + 8);
      buf2 = _mm256_loadu_ps(b + 8);
      c1 = _mm256_mul_ps(buf1, buf2);
    }

    if (accumulators > 2) {
      buf1 = _mm256_loadu_ps(a + 16);
      buf2 = _mm256_loadu_ps(b + 16);
      c2 = _mm256_mul_ps(buf1, buf2);
    }

    if (accumulators > 3) {
      buf1 = _mm256_loadu_ps(a + 24);
      buf2 = _mm256_loadu_ps(b + 24);
      c3 = _mm256_mul_ps(buf1, buf2);
    }

    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_fmadd_ps(buf1, buf2, c0);

    if (accumulators > 1) {
      buf1 = _mm256_loadu_ps(a + 8);
      buf2 = _mm256_loadu_ps(b + 8);
      c1 = _mm256_fmadd_ps(buf1, buf2, c1);
    }

    if (accumulators > 2) {
      buf1 = _mm256_loadu_ps(a + 16);
      buf2 = _mm256_loadu_ps(b + 16);
      c2 = _mm256_fmadd_ps(buf1, buf2, c2);
    }

    if (accumulators > 3) {
      buf1 = _mm256_loadu_ps(a + 24);
      buf2 = _mm256_loadu_ps(b + 24);
      c3 = _mm256_fmadd_ps(buf1, buf2, c3);
    }
  }

  if (accumulators > 1)
    c0 = _mm256_add_ps(c0, c1);
  if (accumulators > 3)
    c2 = _mm256_add_ps(c2, c3);
  if (accumulators > 2)
    c0 = _mm256_add_ps(c0, c2);
  
  float buffer[8];
  _mm256_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

__attribute__((noinline)) float dot_product_sse_multiple_accumulators(float *a, float *b, size_t size, int accumulators) {
  int step = accumulators * 4;
  const float* limit = a + size;
  assert(0 == size % accumulators);

  __m128 c0,c1, c2,c3;

  {
    __m128 buf1 = _mm_loadu_ps(a);
    __m128 buf2 = _mm_loadu_ps(b);
    c0 = _mm_mul_ps(buf1, buf2);

    if (accumulators > 1) {
      buf1 = _mm_loadu_ps(a + 4);
      buf2 = _mm_loadu_ps(b + 4);
      c1 = _mm_mul_ps(buf1, buf2);
    }

    if (accumulators > 2) {
      buf1 = _mm_loadu_ps(a + 8);
      buf2 = _mm_loadu_ps(b + 8);
      c2 = _mm_mul_ps(buf1, buf2);
    }

    if (accumulators > 3) {
      buf1 = _mm_loadu_ps(a + 12);
      buf2 = _mm_loadu_ps(b + 12);
      c3 = _mm_mul_ps(buf1, buf2);
    }
    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m128 buf1 = _mm_loadu_ps(a);
    __m128 buf2 = _mm_loadu_ps(b);
    c0 = _mm_fmadd_ps(buf1, buf2, c0);

    if (accumulators > 1) {
      buf1 = _mm_loadu_ps(a + 4);
      buf2 = _mm_loadu_ps(b + 4);
      c1 = _mm_fmadd_ps(buf1, buf2, c1);
    }

    if (accumulators > 2) {
      buf1 = _mm_loadu_ps(a + 8);
      buf2 = _mm_loadu_ps(b + 8);
      c2 = _mm_fmadd_ps(buf1, buf2, c2);
    }

    if (accumulators > 3) {
      buf1 = _mm_loadu_ps(a + 12);
      buf2 = _mm_loadu_ps(b + 12);
      c3 = _mm_fmadd_ps(buf1, buf2, c3);
    }
  }

  if (accumulators > 1)
    c0 = _mm_add_ps(c0, c1);
  if (accumulators > 3)
    c2 = _mm_add_ps(c2, c3);
  if (accumulators > 2)
    c0 = _mm_add_ps(c0, c2);
  
  float buffer[4];
  _mm_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}

__attribute__((noinline)) float dot_product(float *a, float *b, size_t size) {
  float c = 0;
  const float* limit = a + size;
  for ( ; a < limit; a++, b++) {
    c += a[0] * b[0];
  }
  return c;
}