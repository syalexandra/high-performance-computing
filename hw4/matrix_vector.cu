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

#define N (1UL<<20)
#define M (1UL<<15)

#define BLOCK_SIZE (1UL<<15)
#define GRID_SIZE (1UL<<15)

void vec_mat_product(double* c, const double* a, const double* b){
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
    
        double temp=0;
        for(long j = 0; j < M; j++){
            temp+=a[i*M+j]*b[j];
        }
        
        c[i]=temp;
    }
}



__global__
void vec_mat_product_kernel(double* c,const double* a, const double* b){


    
    __shared__ double temp[M];
    
    int col=threadIdx.x+blockDim.x*blockIdx.x;
    int row=blockIdx.x*gridDim.x+blockIdx.y;
    
    if(col<M){
        temp[col]=a[row*M+col]*b[col];
    }
    
    __syncthreads();
    
    if(0==threadIdx.x){
        int summ=0;
        for(int i=0;i<M;i++){
            summ+=temp[i];
        }
        c[row]=summ;
    }
}




void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
    
    double* x = (double*) malloc(N*M * sizeof(double));
    double* y = (double*) malloc(M * sizeof(double));
    double* z = (double*) malloc(N * sizeof(double));
    double* z_ref = (double*) malloc(N * sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N*M; i++)
    {
      x[i] = i+2;
    }
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < M; i++)
    {
      y[i] = 1/(i+2);
    }
    
    
    
    double *x_d, *y_d, *z_d;
    cudaMalloc(&x_d, N * M * sizeof(double));
    cudaMalloc(&y_d, M * sizeof(double));
    cudaMalloc(&z_d, N * sizeof(double));
    
    
    
    
    
    double tt = omp_get_wtime();
    vec_mat_product(z_ref, x, y);
    printf("CPU Bandwidth = %f GB/s\n", (N+2)*M*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    

    tt = omp_get_wtime();
    
    cudaMemcpy(x_d, x, N*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(N/GRID_SIZE,GRID_SIZE);
    dim3 dimBlock(M/BLOCK_SIZE,BLOCK_SIZE);
    vec_mat_product_kernel<<<dimGrid,dimBlock>>>(z_d,x_d,y_d);
    cudaDeviceSynchronize();
    
    cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU Bandwidth = %f GB/s\n", (N+2)*M*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double err=0;
    for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    printf("error = %f ",err);
    
}
