//
//  main.cpp
//  homework
//
//  Created by Yue Sun on 2/17/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//

#include <iostream>
#include <cmath>
#include "utils.h"
using namespace std;

#define TILE_DIM 32
#define N (1UL<<5)
#define BLOCK_ROWS 8


void jacobian(double * u,double * f){
    int NREPEAT=5000;
    double * v = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double * temp;
    double h=1.0/(N+1);
    double hsquare=h*h;
    
    int i,j;
    
    for(i=0;i<(N+2)*(N+2);i++){v[i]=0;}
    
    double norm;
    double beginNorm=0;
    #pragma omp parallel for collapse(2) reduction (+:beginNorm)
    for(i=1;i<N+1;i++){
        for(j=1;j<N+1;j++){
            beginNorm = beginNorm + f[i*(N+2)+j]*f[i*(N+2)+j];
        }
    }
    
    //cout<<beginNorm<<endl;
    for(int r=0;r<NREPEAT;r++)
    {
        #pragma omp parallel for collapse(2)
        for(i=1;i<=N;i++){
            
            for(j=1;j<=N;j++){
                v[i*(N+2)+j]=(h*h*f[i*(N+2)+j]+u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1])/4.0;
                //u[i*(N+2)+j]=v[i*(N+2)+j];
            }
        }
        
        norm=0;
        #pragma omp parallel for collapse(2) reduction (+:norm)
        for(i=1;i<=N;i++){
            
            for(j=1;j<=N;j++){
                norm+=pow((v[(i-1)*(N+2)+j]+v[i*(N+2)+j-1]+v[(i+1)*(N+2)+j]+v[i*(N+2)+j+1]-4*v[i*(N+2)+j])/hsquare+f[i*(N+2)+j],2);
            }
        }
        
        //cout<<"norm"<<endl;
        //cout<<norm<<endl;
        temp=u;
        u=v;
        v=temp;
        if(sqrt(norm)<=sqrt(beginNorm)/1e6)
            {
                //cout<<"break here"<<r;
                break;
            }
    }
    
    //return u;
}


__global__ void jacobiUpdate(double* x_old,double* x_new,double* f){
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    double h=1.0/(N+1);
    double hsquare=h*h;
    
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        x_new[(y+j)*width + x]=(x_old[(y+j)*width + x-1]+x_old[(y+j)*width + x+1]+x_old[(y+j-1)*width + x]+x_old[(y+j+1)*width + x]+hsquare*f[(y+j)*width + x])/4;
    
}


int main(int argc, char ** argv) {
    // insert code here...
    
    double * f = (double *)malloc((N+2)*(N+2)* sizeof(double));
    for(int i=0;i<(N+2)*(N+2);i++){f[i]=1;}
    double * x=(double*) malloc((N+2)*(N+2)*sizeof(double));
    for(int i=0;i<(N+2)*(N+2);i++){x[i]=0;}
    Timer t;
    t.tic();
    //jacobian(x,f);
    //for(int i=0;i<(N+2)*(N+2);i++)cout<<x[i]<<" ";
    cout<<"openmp time: "<<t.toc()<<endl;
    
    
    /*
    for(int i=0;i<N;i+=100){
        cout<<x[i];
    }
    */
    
    
    //double * x=(double*) malloc((N+2)*(N+2)*sizeof(double));
    double * x_next=(double*) malloc((N+2)*(N+2)*sizeof(double));
    
    for(int i=0;i<(N+2)*(N+2);i++){
        //x[i]=0;
        x_next[i]=0;
    }
    
    
    
    t.tic();
    
    double *x_next_d,*x_d,*f_d;
    
    cudaMalloc((void **) &x_next_d,(N+2)*(N+2)* sizeof(double));
    cudaMalloc((void **) &x_d,(N+2)*(N+2)* sizeof(double));
    cudaMalloc((void **) &f_d,(N+2)*(N+2)* sizeof(double));
    
    cudaMemcpy(x_next_d, x_next, (N+2)*(N+2)* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x, (N+2)*(N+2)* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, (N+2)*(N+2)* sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 GridDim((N+2)/TILE_DIM,(N+2)/TILE_DIM);
    dim3 BlockDim(TILE_DIM, BLOCK_ROWS);
    
    for(int i=0;i<5000;i++){
        if(i%2==0){
            jacobiUpdate<<<GridDim,BlockDim>>>(x_d,x_next_d,f_d);
        }
        else{
            jacobiUpdate<<<GridDim,BlockDim>>>(x_next_d,x_d,f_d);
        }
    }
    
    for(int i=0;i<(N+2)*(N+2);i++)cout<<x_d[i]<<" ";
    cout<<"cuda time: "<<t.toc()<<endl;
    
    free(x);
    free(f);
    free(x_next);
    cudaFree(x_d);
    cudaFree(x_next_d);
    cudaFree(f_d);
    
    return 0;
}
