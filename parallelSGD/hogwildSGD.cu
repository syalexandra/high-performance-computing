//
//  cudaSGD.cpp
//  parallelSGD
//
//  Created by Yue Sun on 4/22/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//


#include "dataReader.h"
#include "PSGD.h"
#include "MultiLog.h"
#include "LossType.h"
#include "hogwild.h"

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

#define BLOCK_SIZE 1024

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


double getLossHogwild(double* weight,double** trainingData,uchar* trainingLabel,int n_data,int n_weights,int n_labels,double lambda){

    double summ=0;
    double* exponent=(double*)malloc(n_labels*sizeof(double));
    for(int i=0;i<n_data;i++){
        
        double expSum=0;
        for(int k=0;k<n_labels;k++){
            exponent[k]=0;
            for(int j=0;j<n_weights;j++){
                exponent[k]+=weight[k*n_weights+j]*trainingData[i][j];
                if(k==trainingLabel[i]){
                    summ-=exponent[k];
                }
            }
            expSum+=exp(exponent[k]);
        }
        if(expSum>0)summ+=log(expSum);
    }
    
    double regularizationTerm=0;
    for(int k=0; k<n_labels; k++){
        for(int j=0; j<n_weights; j++){
            regularizationTerm += weight[k*n_weights+j] * weight[k*n_weights+j];
        }
    }
    
    regularizationTerm*=lambda;
    return regularizationTerm+summ;
}

int main(int argc, const char * argv[]) {
    printf("Started main.\n");
    mnist data;
    int n_images;
    int size_image;
    double **tempData;
    tempData = data.read_mnist_images("train-images.idx3-ubyte",n_images, size_image);
    //n_images 60000,size_image=784
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
   
    int n_images_test;
    int size_image_test;
    double **testingData;
    testingData = data.read_mnist_images("t10k-images-idx3-ubyte", n_images_test, size_image_test);
    int n_labels_test;
    uchar *testingLabels;
    testingLabels = data.read_mnist_labels("t10k-labels-idx1-ubyte",n_labels_test);

    double *trainingData_d;
    uchar  *trainingLabel_d;
    cudaMalloc(&trainingData_d, (n_images * (size_image+1) *sizeof(double)));
    Check_CUDA_Error("Malloc trainingData failed");
    cudaMalloc(&trainingLabel_d, n_images *sizeof(uchar));
    cudaMemcpy(trainingData_d, trainingData, (n_images * (size_image+1) *sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(trainingLabel_d, trainingLabel,n_images *sizeof(uchar), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 
    printf("Data loaded.\n");

    PSGD psgd(1);
    psgd.initialize(size_image+1,10);
    //vector<double> weight_openmp = psgd.getWeight();
    
    
    
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
    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
    
    double* weight_d;
    cudaMalloc(&weight_d, weight_size*sizeof(double));
    cudaMemcpy(weight_d, weight, weight_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();    
    printf("\nGPU:\n");
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) { 
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
    }
    
    printf("Enter iterations (> 10):\n");
    int n_iterations=1000;
    scanf("%d", &n_iterations);
    
    double eta;
    eta=0.001;
    printf("\nEnter learning rate (eta = 0.001):\n");
    //scanf("%lf", &eta);
    
    double lambda;
    lambda=0.001;
    printf("\nEnter regularization parameter (lambda = 0.001):\n");
    //scanf("%lf", &lambda);
    
    double oldLoss=getLoss(weight,tempData,tempLabel,n_images,size_image+1,10,lambda);
    printf("old loss: %f \n",oldLoss);
    double t = omp_get_wtime();

    int n_blocks = 1;
    //update the weight
    for(int j=0;j<n_iterations;j++){
	run_hogwild_one_processor<<<n_blocks, BLOCK_SIZE>>>(weight_d,trainingData_d,trainingLabel_d,eta,n_images,size_image+1,10,lambda, j);
	cudaDeviceSynchronize();
        //printf("Iteration %i done.\n", j);
    	if(j %(n_iterations/5) == 0 || j == n_iterations-1){
          cudaMemcpy(weight, weight_d, weight_size*sizeof(double), cudaMemcpyDeviceToHost);
	  cudaDeviceSynchronize();
          printf("weight[107] = %f\n", weight[107]);
	  double loss_now = getLoss(weight, tempData, tempLabel, n_images, size_image+1,10, lambda);
	  printf("Training (log)loss: %f\t thread:%d\n",loss_now, omp_get_thread_num());
	  psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
	}
    }
    
    t = omp_get_wtime() - t;
    printf("Training ran for %10f\n", t);
    
    cudaMemcpy(weight, weight_d, weight_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double newLoss=getLoss(weight,tempData,tempLabel,n_images,size_image+1,10,lambda);
    printf("new loss: %f \n",newLoss);

    MultiLog mlog;
    mlog.setLambda(lambda);//Regularization parameter
    psgd.loss = &mlog;//setting the loss function in PSGD object

    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
 
    printf("End\n");
    free(tempData);
    free(tempLabel);
    cudaFree(trainingData);
    cudaFree(trainingLabel);
    cudaFree(weight);
	checkCuda( cudaFree(trainingData_d));	
	checkCuda( cudaFree(trainingLabel_d));	
	checkCuda( cudaFree(weight_d));
}

