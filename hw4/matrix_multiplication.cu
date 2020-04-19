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

    __shared__ float chache[BLOCK_SIZE] ;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chacheindex = threadIdx.x;
    double temp;
    
    while ( tid < N )
    {
         temp += a[tid] * b[tid] ;
         tid += blockDim.x * gridDim.x ;
    }
    
    chache[chacheindex] = temp ;

    __synchthreads ();
    int i  = blockDim.x / 2;
    while ( i!=0 )
     {
        if ( chacheindex < i )
            chache[chacheindex] += chache [chacheindex + i] ;
        
        __synchthreads();
        i/=2 ;
     }
     
     if ( chacheindex == 0 )c[blockIdx.x] = chache [0] ;

    
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
    cudaMallocManaged(&s, BLOCK_SIZE * sizeof(double));
    
    //cudaMallocManaged(&s, N * sizeof(double));
    
    double* s_ref;
    
    
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++)
    {
      x[i] = i+2;
      y[i] = 1.0/(i+1);
    }
    
    
    
    double tt = omp_get_wtime();
    vec_inner_product(s_ref,x, y);
    printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    

    tt = omp_get_wtime();
    vec_inner_product_kernel<<<N/1024+1,1024>>>(s,x,y);
    cudaDeviceSynchronize();
    
    float sum = 0;
    for ( int i = 0 ; i<BLOCK_SIZE ; i++ )
        sum+=s[i];
    printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
    
    
    double err = 0;
    //for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
    err=s_ref-sum;
    printf("Error = %f %f %f\n", err,s_ref,sum);
    
    
    
}
