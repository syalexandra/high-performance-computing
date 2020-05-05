// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  
  // sort locally
  std::sort(vec, vec+N);
  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
    
    int root=0;
    
    
    
    int interval=N/(p-1);
    int *sendArray=(int*)malloc((p-1)*sizeof(int));
    
    for(int i=0;i<p-1;i++){
        sendArray[i]=vec[(i+1)*interval-interval/2];
        //printf("sendArray: %d ",sendArray[i]);
    }
    //printf("\n");
    
    
    int* rootBuf;
    int *broadCastArray=(int*)malloc((p-1)*sizeof(int));
    
    if(rank==root){
        rootBuf=(int*)malloc(p*(p-1)*sizeof(int));
    }
    MPI_Gather(sendArray,p-1,MPI_INT,rootBuf,p-1,MPI_INT,root,MPI_COMM_WORLD);
    
    if(rank==root){
    
        std::sort(rootBuf, rootBuf+p*(p-1));
        
        for(int i=0;i<p*(p-1);i++){
            printf("%d ",rootBuf[i]);
        }
        printf("\n");
        
        for(int i=0;i<p-1;i++){
            broadCastArray[i]=rootBuf[(i+1)*p-p/2];
            //printf("broadCastArray %d",broadCastArray[i]);
        }
    }
    
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
    
  // root process broadcasts splitters to all other processes
    
    
    MPI_Bcast(broadCastArray,p-1,MPI_INT,root,MPI_COMM_WORLD);
    
    for(int i=0;i<p-1;i++){
        printf("bcast: %d %d",rank, broadCastArray[i]);
    }
    printf("\n");
    
    
    
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
    
    int* sdispls=(int*)malloc(p*sizeof(int));
    int* scounts=(int*)malloc(p*sizeof(int));
    
    printf("send displacement %d: ",rank);
    for(int i=0;i<p;i++){
        if(i==0){
            sdispls[i]=0;
        }
        else{
            sdispls[i]=std::lower_bound(vec,vec+N,broadCastArray[i-1])-vec;
        }
        printf("%d ",sdispls[i]);
        
    }
    
    printf("send %d: ",rank);
    for(int i=0;i<p;i++){
        if(i==p-1){
            scounts[i]=N-sdispls[i];
        }
        else{
            scounts[i]=sdispls[i+1]-sdispls[i];
        }
        printf("%d ",scounts[i]);
    }
    printf("\n");
    
    
    int* recvcounts=(int*)malloc(p*sizeof(int));
    
    MPI_Alltoall(scounts,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
    
    
    printf("recv %d: ",rank);
    for(int i=0;i<p;i++){
        printf("%d ",recvcounts[i]);
    }
    
    printf("\n");
    
    int* rdispls=(int*)malloc(p*sizeof(int));
    int recv_length=0;
    
    for(int i=0;i<p;i++){
        if(i==0){
            rdispls[i]=0;
        }
        else{
            rdispls[i]=rdispls[i-1]+recvcounts[i-1];
        }
        recv_length+=recvcounts[i];
    }
    printf("rank %d length of receive %d:",rank,recv_length);
    
    int* buffer_recv=(int*)malloc(recv_length*sizeof(int));
    
    MPI_Alltoall(vec,scounts,sdispls,MPI_INT,buffer_recv,recvcounts,rdispls,MPI_INT,MPI_COMM_WORLD);
    
    printf("rank %d : ",rank);
    for(int i=0;i<recv_length;i++){
        printf("%d ",buffer_recv[i]);
    }
    
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  // do a local sort of the received data

  // every process writes its result to a file

  free(vec);
  MPI_Finalize();
  return 0;
}
