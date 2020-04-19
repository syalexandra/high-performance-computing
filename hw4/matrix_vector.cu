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


#define BLOCK_SIZE (1UL<<10)
#define GRID_SIZE (1UL<<10)

void vec_mat_product(double* c, const double* a, const double* b,long N,long M){
    
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
void vec_mat_product_kernel(double* c,const double* a, const double* b,long N,long M){


    
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

    long N= (1UL<<20)
    long M= (1UL<<10)
    
    double *x, *y, *z;
    
    cudaMallocManaged(&x, N * M * sizeof(double));
    cudaMallocManaged(&y, M * sizeof(double));
    cudaMallocManaged(&z, N * sizeof(double));
    
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
    
    
    double tt = omp_get_wtime();
    vec_mat_product(z_ref, x, y, N);
    printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    

    tt = omp_get_wtime();
    dim3 dimGrid(N/GRID_SIZE,GRID_SIZE);
    dim3 dimBlock(M/BLOCK_SIZE,BLOCK_SIZE);
    
    vec_mat_product_kernel<<<dimGrid,dimBlock>>>(z,x,y,N,M);
    cudaDeviceSynchronize();
    
    printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    double err=0;
    for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    
    
}
