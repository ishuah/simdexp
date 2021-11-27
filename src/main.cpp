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

  float aa[8][8] = 
    {
      {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8}, 
      {0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6},
      {1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},
      {2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2},
      {3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0},
      {4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8},
      {4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6},
      {5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4}
      };

  float bb[8][8] = 
    {
      {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7},
      {0.2,1.0,1.8,2.6,3.4,4.2,5.0,5.8},
      {0.3,1.1,1.9,2.7,3.5,4.3,5.1,5.9},
      {0.4,1.2,2.0,2.8,3.6,4.4,5.2,6.0},
      {0.5,1.3,2.1,2.9,3.7,4.5,5.3,6.1},
      {0.6,1.4,2.2,3.0,3.8,4.6,5.4,6.2},
      {0.7,1.5,2.3,3.1,3.9,4.7,5.5,6.3},
      {0.8,1.6,2.4,3.2,4.0,4.8,5.6,6.4}
      };
  
  float cc[8][8] = {};

  dot_product(aa, bb, cc);

  cout << "\n\ndot product naive: \n";
  for (int i = 0; i < 8; i++) {
    cout << "[\t";
    for (int j = 0; j < 8; j++) {
      cout << cc[i][j] << "\t";
    }
    cout << " ]\n";
  }

  return 0;
}
