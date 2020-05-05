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

#define N (1UL<<12)

#define BLOCK_SIZE (1UL<<10)

__global__ void vec_inner_product(double* sum, const double* a, const double* b){
    
    __shared__ double smem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    // each thread reads data from global into shared memory
    if (idx < N) smem[threadIdx.x] = a[idx]*b[idx];
    else smem[threadIdx.x] = 0;
    __syncthreads();

    // x >>= 1 means "set x to itself shifted by one bit to the right", i.e., a divison by 2
    // write to memory with threadIdx rather than ``index''
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (threadIdx.x < s) {
          smem[threadIdx.x] += smem[threadIdx.x + s];
      }
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
    
    double* x = (double*) malloc(N * sizeof(double));
    double* y = (double*) malloc(N * sizeof(double));
    
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    
    double* z = (double*) malloc(Nb * sizeof(double));
        
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++)
    {
      x[i] = i+2;
    }
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++)
    {
      y[i] = 1/(i+2);
    }
    
    
    
    double *x_d, *y_d, *z_d;
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_d, N * sizeof(double));
    cudaMalloc(&z_d, Nb * sizeof(double));
    

    double tt = omp_get_wtime();
    
    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    
    
    vec_inner_product<<<Nb,BLOCK_SIZE>>>(z_d,x_d,y_d);
    
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(z, z_d, Nb*sizeof(double), cudaMemcpyDeviceToHost);
    double summ=0;
    for(int i=0;i<Nb;i++){
        summ+=z_d[i];
    }
    printf("sum: %f",summ);
    printf("GPU Bandwidth = %f GB/s\n", (N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    
}
