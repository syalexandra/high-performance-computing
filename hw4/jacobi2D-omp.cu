//
//  main.cpp
//  homework
//
//  Created by Yue Sun on 2/17/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//

#include <iostream>
#include <math.h>
#include "utils.h"
using namespace std;

#define BLOCK_SIZE (1UL<<3)
#define N (1UL<<5)-2


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
        //cout<<sqrt(norm)<<endl;
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
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double h=1.0/(N+1);
    double hsquare=h*h;
    //printf("x:%d,%d,%d,%d\n",blockIdx.x,blockDim.x,threadIdx.x,x);
    //printf("y:%d,%d,%d,%d\n",blockIdx.y,blockDim.y,threadIdx.y,y);
    
    if(x>0 && y>0 && x<=N && y<=N){
        x_new[x* (N+2)+ y]=(x_old[(x-1)* (N+2)+ y]+x_old[(x+1)*(N+2)+ y]+x_old[x*(N+2)+ y-1]+x_old[x*(N+2)+ y+1]+hsquare*f[x*(N+2)+y])/4.0;
    }
    
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
    
    
    
    double * x_next=(double*) malloc((N+2)*(N+2)*sizeof(double));
    
    for(int i=0;i<(N+2)*(N+2);i++){
        x[i]=0;
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
    
    dim3 GridDim((N+2)/BLOCK_SIZE,(N+2)/BLOCK_SIZE);
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE);
    double h=1.0/(N+1);
    double hsquare=h*h;
    
    for(int i=0;i<10;i++){
        if(i%2==0){
            jacobiUpdate<<<GridDim,BlockDim>>>(x_d,x_next_d,f_d);
            cudaMemcpy(x, x_d, (N+2)*(N+2)* sizeof(double), cudaMemcpyDeviceToHost);
            /*
            for(int i=0;i<=N+1;i++){
                for(int j=0;j<=N+1;j++){
                    printf("%f ",x[i*(N+2)+j]);
                }
                printf("\n");
            }
            */
            
            double norm=0;
            //#pragma omp parallel for collapse(2) reduction (+:norm)
            for(int i=1;i<=N;i++){
                
                for(int j=1;j<=N;j++){
                    norm+=pow((x[(i-1)*(N+2)+j]+x[i*(N+2)+j-1]+x[(i+1)*(N+2)+j]+x[i*(N+2)+j+1]-4*x[i*(N+2)+j])/hsquare+f[i*(N+2)+j],2);
                }
            }
            
            printf("norm = %f \n",sqrt(norm));
            
        }
        else{
            jacobiUpdate<<<GridDim,BlockDim>>>(x_next_d,x_d,f_d);
            
            cudaMemcpy(x_next, x_next_d, (N+2)*(N+2)* sizeof(double), cudaMemcpyDeviceToHost);
            /*
            for(int i=0;i<=N+1;i++){
                for(int j=0;j<=N+1;j++){
                    printf("%f ",x_next[i*(N+2)+j]);
                }
                printf("\n");
            }
            */
            
            double norm=0;
            //#pragma omp parallel for collapse(2) reduction (+:norm)
            for(int i=1;i<=N;i++){
                
                for(int j=1;j<=N;j++){
                    norm+=pow((x_next[(i-1)*(N+2)+j]+x_next[i*(N+2)+j-1]+x_next[(i+1)*(N+2)+j]+x_next[i*(N+2)+j+1]-4*x_next[i*(N+2)+j])/hsquare+f[i*(N+2)+j],2);
                }
            }
            
            printf("norm = %f \n",sqrt(norm));
            
        }
    }
    
    cout<<"cuda time: \n"<<t.toc()<<endl;
    
    //cudaMemcpy(x, x_d, (N+2)*(N+2)* sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(x_next, x_next_d, (N+2)*(N+2)* sizeof(double), cudaMemcpyDeviceToHost);
    
    
    
    free(x);
    free(f);
    free(x_next);
    cudaFree(x_d);
    cudaFree(x_next_d);
    cudaFree(f_d);
    
    return 0;
}
