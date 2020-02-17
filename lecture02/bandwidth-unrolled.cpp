// Measuring bandwidth from different cache levels
// $ g++ -std=c++11 -O3 -march=native bandwidth-unrolled.cpp && ./a.out -n 400000000

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  long repeat = 1e9 / n;

  double* x = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) x[i] = i+1;

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = k;
    for (long i = 0; i < n; i += 8) {
      x[i+ 0] = kk;
      x[i+ 1] = kk;
      x[i+ 2] = kk;
      x[i+ 3] = kk;
      x[i+ 4] = kk;
      x[i+ 5] = kk;
      x[i+ 6] = kk;
      x[i+ 7] = kk;
    }
  }
  printf("time to write          = %8.2f s    ", t.toc());
  printf("bandwidth = %8.2f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = x[0] * 0.5;
    for (long i = 0; i < n; i += 8) {
      x[i+ 0] = x[i+ 0] * 0.5 + kk;
      x[i+ 1] = x[i+ 1] * 0.5 + kk;
      x[i+ 2] = x[i+ 2] * 0.5 + kk;
      x[i+ 3] = x[i+ 3] * 0.5 + kk;
      x[i+ 4] = x[i+ 4] * 0.5 + kk;
      x[i+ 5] = x[i+ 5] * 0.5 + kk;
      x[i+ 6] = x[i+ 6] * 0.5 + kk;
      x[i+ 7] = x[i+ 7] * 0.5 + kk;
    }
  }
  printf("time to read + write   = %8.2f s    ", t.toc());
  printf("bandwidth = %8.2f GB/s\n", 2 * n * repeat * sizeof(double) / 1e9 / t.toc());

  t.tic();
  double sum[8] = {0};
  for (long k = 0; k < repeat; k++) {
    for (long i = 0; i < n; i += 8) {
      sum[ 0] += x[i+ 0];
      sum[ 1] += x[i+ 1];
      sum[ 2] += x[i+ 2];
      sum[ 3] += x[i+ 3];
      sum[ 4] += x[i+ 4];
      sum[ 5] += x[i+ 5];
      sum[ 6] += x[i+ 6];
      sum[ 7] += x[i+ 7];
    }
  }
  double sum_ = 0;
  for (int i = 0; i < 8; i++) sum_ += sum[i];
  printf("time to read           = %8.2f s    ", t.toc());
  printf("bandwidth = %8.2f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());

  free(x);

  return sum_;
}

