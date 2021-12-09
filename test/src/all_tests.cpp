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

  virtual void verify_dot_product_avx2() {
    float a[16] = {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7};
    float b[16] = {0.1,0.9,1.7,2.5,3.3,4.1,4.9,5.7,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8};
    float c;
    size_t size = 16;

    c = dot_product_avx2(a, b, size);
    EXPECT_FLOAT_EQ(c, 27.6);
  }

  virtual void verify_dot_product_avx2_two_accumulators() {
    size_t size = 32;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    c = dot_product_avx2_two_accumulators(a, b, size);
    EXPECT_FLOAT_EQ(c, 59.84);
  }

  virtual void verify_dot_product_avx2_four_accumulators() {
    size_t size = 32;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    c = dot_product_avx2_four_accumulators(a, b, size);
    EXPECT_FLOAT_EQ(c, 59.84);
  }

  virtual void verify_dot_product_avx2_multiple_accumulators() {
    size_t size = 32;
    float a[size];
    float b[size];
    float c;
    
    fill_vectors(a, b, size);

    c = dot_product_avx2_multiple_accumulators(a, b, size, 2);
    EXPECT_FLOAT_EQ(c, 59.84);
    c = 0.0;
    c = dot_product_avx2_multiple_accumulators(a, b, size, 4);
    EXPECT_FLOAT_EQ(c, 59.84);
  }

  virtual void verify_dot_product() {
    size_t size = 32;
    float a[size];
    float b[size];
    float c;

    fill_vectors(a, b, size);

    c = dot_product(a, b, size);
    EXPECT_FLOAT_EQ(c, 59.84);
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

TEST_F(SimdexpTest, dot_product_avx2) {
  verify_dot_product_avx2();
}

TEST_F(SimdexpTest, dot_product) {
  verify_dot_product();
}

TEST_F(SimdexpTest, dot_product_avx2_two_accumulators) {
  verify_dot_product_avx2_two_accumulators();
}

TEST_F(SimdexpTest, dot_product_avx2_four_accumulators) {
  verify_dot_product_avx2_four_accumulators();
}

TEST_F(SimdexpTest, dot_product_avx2_multiple_accumulators) {
  verify_dot_product_avx2_multiple_accumulators();
}