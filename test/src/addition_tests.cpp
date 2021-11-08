#include <addition.h>
#include <iostream>
#include "gtest/gtest.h"

using namespace std;

class SimdexpTest : public ::testing::Test {

protected:
  virtual void SetUp() {
  };

  virtual void TearDown() {
  };

  virtual void verify() {
    float a[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float b[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float c[8] = {};
    add_float32(a, b, c);
    EXPECT_FLOAT_EQ(c[0], 0.2);
  }
};

TEST_F(SimdexpTest, a_plus_b) {
  verify();
}
