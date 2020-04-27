//
//  hogwild.cpp
//  parallelSGD
//
//  Created by Yue Sun on 4/22/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//


#include "dataReader.h"


#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <chrono>
#include <omp.h>
#include <vector>
#include <curand_kernel.h>
#include <curand.h>

using namespace std;
typedef unsigned char uchar;

__host__ __device__ double getOneGradient(double* weight,int index,const double*trainingData,const uchar* trainingLabel,double eta,int n_data,int n_weights,int n_labels,double lambda){
    
    double delta_weight=0;
    int i=index / n_weights;//i is for label i
    int j=index % n_weights;//j is for data j
    //printf("%d %d %d \n",i,j,n_data);
    
    for(int b=0;b<n_data;b++){
    
        double probSum=0;
        double* probList;
        probList=(double*)malloc(n_labels*sizeof(double));
        
        for(int l=0;l<n_labels;l++){
            double probExp=0;
            for(int w=0;w<n_weights;w++){
                //printf("%f %f \n",weight[l*n_weights+w],trainingData[b*n_weights+w]);
                probExp+=weight[l*n_weights+w]*trainingData[b*n_weights+w];
            }
            //printf("%f ",probExp);
            probList[l]=exp(probExp);
            probSum+=exp(probExp);
        }
        
        //printf("%f ",probSum);
        probList[i]/=probSum;
        
        
        double sign = (trainingLabel[b]==i)?1:0;
        double partialDerivative = (sign-probList[i])*trainingData[b*n_weights+j];
        partialDerivative += lambda * 2 * weight[i*n_weights+j];
        delta_weight -= partialDerivative;
        
    }
    //printf("delta_weight %f",delta_weight);
    return delta_weight;
    
}

__global__ void updateWeightKernel(double* weight,const double* trainingData,const uchar* trainingLabel,double eta,int n_data,int n_weights,int n_labels,int batchSize,double lambda){
    
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int weight_size=n_weights*n_labels;
    
    
    if(index<weight_size){
        double deltaWeight;
        double* data;
        //cudaMalloc(&data,batchSize*n_weights*sizeof(double));
        data=(double*)malloc(batchSize*n_weights*sizeof(double));
        
        uchar* label;
        //cudaMalloc(&label,batchSize*sizeof(uchar));
        label=(uchar*)malloc(batchSize*sizeof(uchar));
        
        for(int b=0;b<batchSize;b++){
            curandState_t state;
            curand_init(index,0,b,&state);
            int r;
            r=curand(&state)%n_data;
            //cudaMemcpy(data,trainingData+r*n_weights,n_weights*sizeof(double),cudaMemcpyHostToDevice);
            //cudaMemcpy(label,trainingLabel+r,sizeof(uchar),cudaMemcpyHostToDevice);
            label[b]=trainingLabel[r];
            //printf("%d ",label[b]);
            for(int w=0;w<n_weights;w++){
                data[b*n_weights+w]=trainingData[r*n_weights+w];
                printf("%f ",data[b*n_weights+w]);
            }
        }
        
        deltaWeight=getOneGradient(weight,index, data, label,eta, batchSize, n_weights, n_labels,lambda/batchSize);
        weight[index]-=eta* deltaWeight;
    }
    
}



int main(int argc, const char * argv[]) {
    // insert code here...
    mnist data;
    int n_images;
    int size_image;
    double **tempData;
    //trainingData = data.read_mnist_images("train-images-idx3-ubyte",n_images, size_image);
    tempData = data.read_mnist_images("train-images.idx3-ubyte",n_images, size_image);
    //n_images 60000,size_images=785
    double * trainingData;
    cudaMallocHost((void**)&trainingData,n_images*(size_image+1)*sizeof(double));
    
    for(int i=0;i<n_images;i++){
        for(int j=0;j<size_image+1;j++){
            trainingData[i*(size_image+1)+j]=tempData[i][j];
        }
    }
    
    int n_labels;
    uchar *tempLabel;
    
    tempLabel = data.read_mnist_labels("train-labels.idx1-ubyte",n_labels);
    uchar* trainingLabel;
    cudaMallocHost((void**)&trainingLabel,n_labels*sizeof(uchar));
    for(int i=0;i<n_labels;i++){
        trainingLabel[i]=tempLabel[i];
    }
    
    int gridSize=size_image+1;
    int blockSize=2;
    
    
    double* weight;
    int weight_size=(size_image+1)*10;
    cudaMallocHost((void**)&weight,weight_size*sizeof(double));
    
    //initialize the weight
    int seed =1;//chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);
    for (int i=0;i<weight_size;i++){
        weight[i]=distribution(generator);
    }
    
    printf("Enter iterations (> 10):\n");
    int n_iterations;
    //scanf("%d", &n_iterations);
    n_iterations=10;
    double eta;
    eta=0.001;
    printf("\nEnter learning rate (eta = 0.001):\n");
    //scanf("%lf", &eta);
    
    double lambda;
    lambda=0.001;
    printf("\nEnter regularization parameter (lambda = 0.001):\n");
    
    //update the weight
    for(int j=0;j<n_iterations;j++){
        
        updateWeightKernel<<<gridSize,blockSize>>>(weight,trainingData,trainingLabel,eta,n_images,size_image+1,10,2,lambda);
        cudaDeviceSynchronize();
        
    }
    
    
    
    
}
