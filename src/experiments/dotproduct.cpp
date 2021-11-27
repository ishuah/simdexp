# include "dotproduct.h"

const int ROWS = 8;
const int COLS = 8;

void dot_product(float a[8][8], float b[8][8], float c[8][8]) {
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      for (int p = 0; p < ROWS; p++) {
        c[i][j] += a[i][p] * b[p][j];
      }
    }
  }
}
