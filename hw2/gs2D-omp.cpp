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

void GaussSeidel(double * u,long N, double * f){
    int NREPEAT=5000;
    //double * v = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double * temp;
    double h=1.0/(N+1);
    double hsquare=h*h;
    long i,j,k=0;
    //for(i=0;i<(N+2)*(N+2);i++){v[i]=u[i];}
    
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
        #pragma omp parallel for
        for(k=1;k<=N;k++){
            long floor=2*k-N>1?2*k-N:1;
            long ceiling=min(N,2*k-1);
            
            for(i=floor;i<=ceiling;i++){
                //v[i*(N+2)+j]=(h*h*f[i*(N+2)+j]+u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1])/4.0;
                //u[i*(N+2)+j]=v[i*(N+2)+j];
                j=2*k-i;
                //cout<<i<<","<<j<<endl;
                u[i*(N+2)+j]=(h*h*f[i*(N+2)+j]+u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1])/4.0;
                
            }
        }
        
        #pragma omp parallel for
        for(k=1;k<=N;k++){
            long floor=2*k-N-1>1?2*k-N-1:1;
            long ceiling=min(N,2*k-2);
            
            for(i=floor;i<=ceiling;i++){
                //v[i*(N+2)+j]=(h*h*f[i*(N+2)+j]+u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1])/4.0;
                //u[i*(N+2)+j]=v[i*(N+2)+j];
                j=2*k-1-i;
                //cout<<i<<","<<j<<endl;
                u[i*(N+2)+j]=(h*h*f[i*(N+2)+j]+u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1])/4.0;
            }
        }
        
        
        
        
        norm=0;
        #pragma omp parallel for collapse(2) reduction (+:norm)
        for(i=1;i<=N;i++){
            for(j=1;j<=N;j++){
                norm+=pow((u[(i-1)*(N+2)+j]+u[i*(N+2)+j-1]+u[(i+1)*(N+2)+j]+u[i*(N+2)+j+1]-4*u[i*(N+2)+j])/hsquare+f[i*(N+2)+j],2);
            }
        }
        
        //cout<<norm<<endl;
        if(sqrt(norm)<=sqrt(beginNorm)/1e6)
            {
                //cout<<"break here"<<r;
                break;
            }
    }
    
}



int main(int argc, char ** argv) {
    // insert code here...
    
    long N = read_option<long>("-n", argc, argv);
    double * f = (double *)malloc((N+2)*(N+2)* sizeof(double));
    for(int i=0;i<(N+2)*(N+2);i++){f[i]=1;}
    double * x=(double*) malloc((N+2)*(N+2)*sizeof(double));
    for(int i=0;i<(N+2)*(N+2);i++){x[i]=0;}
    Timer t;
    t.tic();
    GaussSeidel(x,N,f);
    //for(int i=0;i<(N+2)*(N+2);i++)cout<<x[i]<<" ";
    cout<<"Jacobian Algorithm: "<<t.toc()<<endl;
    
    
    /*
    for(int i=0;i<N;i+=100){
        cout<<x[i];
    }
    */
    free(x);
    free(f);
    return 0;
}
