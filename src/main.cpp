#include <addition.h>
#include <multiplication.h>
#include <dotproduct.h>
#include <iostream>

using namespace std;

static const char *const HEADER = "\nSIMD Experiments\nContol output\n\n";

int main(int argc, const char *argv[]) {
  cout << HEADER;

  float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float c[8] = {};
  float d[8] = {};

  float e[4] = {0.1, 0.2, 0.3, 0.4};
  float f[4] = {0.5, 0.6, 0.7, 0.8};
  float g[4] = {};
  
  
  add_float32_avx2(a, b, c);
  for (float i: c)
    cout << i << '\t';
  cout << '\n';

  add_float32_SIMD(e, f, g);
  for (float i: g)
    cout << i << '\t';
  cout << '\n';
  
  add_float32(a, b, d);
  for (float i: d)
    cout << i << '\t';
  cout << '\n';

  mul_float32_avx2(a, b, c);
  for (float i: c)
    cout << i << '\t';
  cout << '\n';

  mul_float32_SIMD(e, f, g);
  for (float i: g)
    cout << i << '\t';
  cout << '\n';
  
  mul_float32(a, b, d);
  for (float i: d)
    cout << i << '\t';
  cout << '\n';

  size_t size = 786432;
  float aa[size];
  float bb[size];

  fill_vectors(aa, bb, size);

  dot_product(aa, bb, size);
  dot_product_avx2(aa, bb, size);
  dot_product_avx2_multiple_accumulators(aa, bb, size, 2);
  cout << "\n\nsize: " << size;
  return 0;
}
