//
//  matrix_multiplication.c
//  
//
//  Created by Yue Sun on 4/18/20.
//

#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define N (1UL<<25)
#define BLOCK_SIZE 1024


void vec_inner_product(double* c, const double* a, const double* b){
    double temp=0;
    #pragma omp parallel for shared(temp)
    for (long i = 0; i < N; i++) {
        temp += a[i] * b[i];
    }
    
    *c=temp;
    
}



__global__
void vec_inner_product_kernel(double* c,const double* a, const double* b){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
    
    /*
    __syncthreads();
    if(0==idx){
        for(int i=0;i<N;i++){
            *sum+=c[i];
        }
    }
    */
}


__global__
void reduction_kernel0(double* sum, const double* a, long n){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < n) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  for(int s = 1; s < blockDim.x; s *= 2) {
      if(threadIdx.x % (2*s) == 0)
          smem[threadIdx.x] += smem[threadIdx.x + s];
      __syncthreads();
  }

  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}




void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {

    double *x, *y, *z, *s;
    cudaMallocManaged(&x, N * sizeof(double));
    cudaMallocManaged(&y, N * sizeof(double));
    cudaMallocManaged(&z, N * sizeof(double));
    //cudaMallocManaged(&s, N * sizeof(double));
    
    double* s_ref;
    
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++)
    {
      x[i] = i+2;
      y[i] = 1.0/(i+1);
      z[i] = 0;
    }
    
    
    
    double tt = omp_get_wtime();
    vec_inner_product(s_ref,x, y);
    printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    
    

    tt = omp_get_wtime();
    //vec_inner_product_kernel<<<N/1024+1,1024>>>(z,x, y);
    //cudaDeviceSynchronize();
    
    
    long N_work = 1;
    
    for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
    
    //printf("%ld",N_work);
    /*
    cudaMalloc(&s, N_work*sizeof(double));
    
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    
    reduction_kernel0<<<Nb,BLOCK_SIZE>>>(s, z, N);
    
    while (Nb > 1) {
      long N_temp = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel0<<<Nb,BLOCK_SIZE>>>(s + N_temp, s, N_temp);
      s += N_temp;
    }
    
    
    printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    
    //double err = 0;
    //for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    //err=s_ref-s;
    //printf("Error = %f %f %f\n", err,s_ref,s);
    
    */
    
}
