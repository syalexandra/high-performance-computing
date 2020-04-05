#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define nthreads 8
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
    if (n == 0) return;
    prefix_sum[0] = 0;
    omp_set_num_threads(nthreads);
    int threadLen=n/nthreads;
    long*sumList= (long*) malloc((nthreads+1) * sizeof(long));
       
    #pragma omp parallel for
    for(long i=0;i<nthreads;i++)
    {
        long index;
        prefix_sum[i*threadLen]=0;
        for(long j =1;j<threadLen;j++){
            index=i*threadLen+j;
            prefix_sum[index]=prefix_sum[index-1]+A[index-1];
        }
        sumList[i+1]=prefix_sum[index]+A[index];
    }
    
    for(long i=1;i<nthreads;i++){
        sumList[i]=sumList[i]+sumList[i-1];
    }
    
    
    #pragma omp parallel for
    for(long i=0;i<nthreads;i++){
        for(long j=0;j<threadLen;j++){
            long index=i*threadLen+j;
            prefix_sum[index]+=sumList[i];
        }
    }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = i;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
