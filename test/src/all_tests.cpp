#include <addition.h>
#include <multiplication.h>
#include <dotproduct.h>
#include <iostream>
#include "gtest/gtest.h"

using namespace std;

class SimdexpTest : public ::testing::Test {

protected:
  virtual void SetUp() {
  };

  virtual void TearDown() {
  };

  virtual void verify_add_float32_avx2() {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    add_float32_avx2(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.2);
  };

  virtual void verify_add_float32_SIMD() {
    float a[4] = {0.1, 0.2, 0.3, 0.4};
    float b[4] = {0.5, 0.6, 0.7, 0.8};
    float c[4] = {};
    add_float32_SIMD(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.6);
  };

  virtual void verify_add_float32() {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    add_float32(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.2);
  }

    virtual void verify_mul_float32_avx2() {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    mul_float32_avx2(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.01);
  };

  virtual void verify_mul_float32_SIMD() {
    float a[4] = {0.1, 0.2, 0.3, 0.4};
    float b[4] = {0.5, 0.6, 0.7, 0.8};
    float c[4] = {};
    mul_float32_SIMD(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.05);
  };

  virtual void verify_mul_float32() {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    mul_float32(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.01);
  }

  virtual void verify_dot_product() {
    float a[8][8] = {
      {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8}, 
      {0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6},
      {1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4},
      {2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2},
      {3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0},
      {4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8},
      {4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6},
      {5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4}
      };
    float b[8][8] = {
      {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7},
      {0.2,1.0,1.8,2.6,3.4,4.2,5.0,5.8},
      {0.3,1.1,1.9,2.7,3.5,4.3,5.1,5.9},
      {0.4,1.2,2.0,2.8,3.6,4.4,5.2,6.0},
      {0.5,1.3,2.1,2.9,3.7,4.5,5.3,6.1},
      {0.6,1.4,2.2,3.0,3.8,4.6,5.4,6.2},
      {0.7,1.5,2.3,3.1,3.9,4.7,5.5,6.3},
      {0.8,1.6,2.4,3.2,4.0,4.8,5.6,6.4}
      };
    float c[8][8] = {};

    dot_product(a, b, c);
    EXPECT_FLOAT_EQ(c[0][0], 2.04);
    EXPECT_FLOAT_EQ(c[1][1], 12.92);
    EXPECT_FLOAT_EQ(c[4][4], 107.0);
  }
};

TEST_F(SimdexpTest, add_float32_avx2) {
  verify_add_float32_avx2();
}

TEST_F(SimdexpTest, add_float32_SIMD) {
  verify_add_float32_SIMD();
}

TEST_F(SimdexpTest, add_float32) {
  verify_add_float32();
}

TEST_F(SimdexpTest, mul_float32_avx2) {
  verify_mul_float32_avx2();
}

TEST_F(SimdexpTest, mul_float32_SIMD) {
  verify_mul_float32_SIMD();
}

TEST_F(SimdexpTest, mul_float32) {
  verify_mul_float32();
}

TEST_F(SimdexpTest, dot_product) {
  verify_dot_product();
}