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

  float aa[16] = {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7};
  float bb[16] = {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8};
  float cc;
  size_t size = 16;

  float h = dot_product(aa, bb, size);
  float h2 = dot_product_avx2(aa, bb, size);
  float h3 = dot_product_avx2_two_accumulators(aa, bb, size);
  cout << "\n\ndot product naive: " << h;
  cout << "\ndot product avx2 fma:\t" << h2;
  cout << "\ndot product avx2 fma multiple accumulators:\t" << h3 << "\n";
  return 0;
}
