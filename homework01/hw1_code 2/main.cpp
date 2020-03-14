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

double * jacobian(long N, double * f){
    int NREPEAT=5000;
    double * x = (double*) malloc(N* sizeof(double));
    double * y = (double*) malloc(N* sizeof(double));
    
    double hsquare=1.0/((N+1)*(N+1));
    int i;
    for(i=0;i<N;i++){x[i]=0;}
    
    double norm;
    double beginNorm=0;
    for(i=0;i<N;i++){
        beginNorm = beginNorm + f[i] * f[i];
    }
    //cout<<sqrt(beginNorm)<<endl;
    for(int r=0;r<NREPEAT;r++)
    {
        y[0]=f[0]*hsquare/2.0 + x[1]/2.0;
        for(i=1;i<N-1;i++)
        {
            y[i] = f[i]*hsquare/2.0 + x[i-1]/2.0 + x[i+1]/2.0;
        }
        y[N-1]=f[N-1]*hsquare/2.0 + x[N-2]/2.0;
        
        for(i=0;i<N;i++){
            x[i]=y[i];
        }
        
        norm=pow(2.0/hsquare*x[0]-1.0/hsquare*x[1]-f[0],2);
        for(i=1;i<N-1;i++){
            norm+=pow(2.0/hsquare*x[i]-1.0/hsquare*x[i-1]-1.0/hsquare*x[i+1]-f[i],2);
        }
        norm+=pow(2.0/hsquare*x[N-1]-1.0/hsquare*x[N-2]-f[N-1],2);
        
        //cout<<sqrt(norm)<<endl;
        if(sqrt(norm)<=sqrt(beginNorm)/1e6)
            {
                //cout<<"break here"<<r;
                break;
            }
    }
    
    free(y);
    return x;
}



double * GaussSeidel(long N, double * f){
    int NREPEAT=5000;
    double * x = (double*) malloc(N* sizeof(double));
    double hsquare=1.0/((N+1)*(N+1));
    int i;
    for(i=0;i<N;i++){x[i]=0;}
    
    double norm;
    double beginNorm=0;
    for(i=0;i<N;i++){
        beginNorm = beginNorm + f[i] * f[i];
    }
    //cout<<sqrt(beginNorm)<<endl;
    for(int r=0;r<NREPEAT;r++)
    {
        x[0]=f[0]*hsquare/2.0 + x[1]/2.0;
        for(i=1;i<N-1;i++)
        {
            x[i] = f[i]*hsquare/2.0 + x[i-1]/2.0 + x[i+1]/2.0;
        }
        x[N-1]=f[N-1]*hsquare/2.0 + x[N-2]/2.0;
        
        
        norm=pow(2.0/hsquare*x[0]-1.0/hsquare*x[1]-f[0],2);
        for(i=1;i<N-1;i++){
            norm+=pow(2.0/hsquare*x[i]-1.0/hsquare*x[i-1]-1.0/hsquare*x[i+1]-f[i],2);
        }
        norm+=pow(2.0/hsquare*x[N-1]-1.0/hsquare*x[N-2]-f[N-1],2);
        
        //cout<<sqrt(norm)<<endl;
        if(sqrt(norm)<=sqrt(beginNorm)/1e6)
            {
                //cout<<"break here"<<r;
                break;
            }
    }
    
    
    
    
    return x;
}



int main(int argc, char ** argv) {
    // insert code here...
    
    long N = read_option<long>("-n", argc, argv);
    double * f = (double *)malloc(N* sizeof(double));
    for(int i=0;i<N;i++){f[i]=1;}
    double * x;
    Timer t;
    t.tic();
    x = jacobian(N,f);
    cout<<"Jacobian Algorithm: "<<t.toc()<<endl;
    
    t.tic();
    x = GaussSeidel(N,f);
    cout<<"Gauss Seidel Algorithm: "<<t.toc()<<endl;
    
    /*
    for(int i=0;i<N;i+=100){
        cout<<x[i];
    }
    */
    free(x);
    free(f);
    return 0;
}
