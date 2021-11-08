#include <addition.h>
#include <iostream>

using namespace std;

static const char *const HEADER = "\nSIMD Experiments\n\n";

int main(int argc, const char *argv[]) {
  cout << HEADER;

  float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  float c[8] = {};
  float d[8] = {};
  
  add_float32_simd(a, b, c);
  cout << c[0] << "\t" << c[7] << "\n";
  
  add_float32(a, b, d);
  cout << d[0] << "\t" << d[7] << "\n";

  return 0;
}
