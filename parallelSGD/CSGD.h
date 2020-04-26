//
//  HSGD.h
//  parallelSGD
//
//  Created by Yue Sun on 4/22/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//


#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <chrono>
#include "LossType.h"
#include <omp.h>

#ifndef CSGD_h
#define CSGD_h
using namespace std;
typedef unsigned char uchar;

class CSGD{
private:
    int gridSize;
    int blockSize;
    double* weight;
    int weight_size;
    
public:
    CSGD(int gSize,int bSize){
        gridSize=gSize;
        blockSize=bSize;
    }
    
    void initWeight(int n_weights,int n_labels){
        
        int seed =1;//chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator (seed);
        normal_distribution<double> distribution (0.0,1.0);
        
        weight_size=n_weights*n_labels;
        weight=(double*)malloc(weight_size*sizeof(double));
        
        for (int i=0;i<weight_size;i++){
            weight[i]=distribution(generator);
        }
        
        
    }
    
    double* getWeight(){
        return weight;
    }
    
    void updateWeight(LossType& loss,double** trainingData,uchar* testingData,double eta,int n_iterations,int n_data,int n_weights,int n_labels){
        
        double* weight_d;
        cudaMalloc(&weight_d,weight_size*sizeof(double));
        cudaMemcpy(weight_d,weight,weight_size*sizeof(double),cudaMemcpyHostToDevice);
        
        double* train_d;
        cudaMalloc(&)
        uchar test_d;
        
        for(int j=0;j<n_iterations;j++){
            
            
            CSGDKernel<<<gridSize,blockSize>>>(loss,weight_d,traningData,testingData,eta,n_data,n_weights,n_labels);
            cudaDeviceSynchronize();
            
        }
        
        
        cudaMemcpy(weight,weight_d,weight_size*sizeof(double),cudaMemcpyDeviceToHost);
        
    }
    
    ~HSGD(){}
    
};

__global__ void CSGDKernel(LossType& loss,double*weight,const double** trainingData,const uchar* testingDate,double eta,int n_data,int n_weights,int n_labels){
    
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    
    double deltaWeight;
    
    if(index<n_weights*n_labels){
        deltaWeight=loss.getOneGradient(weight,index, trainingData, testingData, n_data, n_weights, n_labels);
        weight[index]-=eta* deltaWeight;
    }
    
}

#endif /* CSGD_h */
