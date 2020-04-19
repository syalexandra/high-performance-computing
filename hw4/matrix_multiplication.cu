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


double vec_inner_product(const double* a, const double* b){
    double temp=0;
    #pragma omp parallel for shared(temp)
    for (long i = 0; i < N; i++) {
        temp += a[i] * b[i];
    }
    return temp;
    
}



__global__
double vec_inner_product_kernel(double* c, const double* a, const double* b){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
    
    __syncthreads();
    if(0==idx){
        int sum=0;
        for(int i=0;i<N;i++){
            sum+=temp[i];
        }
    }
    return sum;
    
}




void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {

    double *x, *y, *z;
    cudaMallocManaged(&x, N * sizeof(double));
    cudaMallocManaged(&y, N * sizeof(double));
    cudaMallocManaged(&z, N * sizeof(double));
    
    //double* z_ref = (double*) malloc(N*sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++)
    {
      x[i] = i+2;
      y[i] = 1.0/(i+1);
      z[i] = 0.0;
    }
    
    
    double tt = omp_get_wtime();
    double s_ref=vec_inner_product(x, y);
    printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    tt = omp_get_wtime();
    double s=vec_inner_product_kernel<<<N/1024+1,1024>>>(z, x, y);
    cudaDeviceSynchronize();
    printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    
    double err = 0;
    //for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    error=s_ref-s;
    printf("Error = %f\n", err);
}
