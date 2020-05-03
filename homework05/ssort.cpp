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
  int N = 10;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  /*
  printf("sort start");
  // sort locally
  std::sort(vec, vec+N);
  printf("sort finish");
  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
    
    int root=0;
    
    
    
    int interval=N/(p-1);
    int *sendArray=(int*)malloc((p-1)*sizeof(int));
    
    for(int i=0;i<p-1;i++){
        sendArray[i]=vec[(i+1)*interval-1];
        printf("sendArray: %d ",sendArray[i]);
    }
    
    
    
    int *broadCastArray=(int*)malloc((p-1)*sizeof(int));
    if(rank==root){
        int* rootBuf=(int*)malloc(p*(p-1)*sizeof(int));
        MPI_Gather(sendArray,p-1,MPI_INT,rootBuf,p-1,MPI_INT,root,MPI_COMM_WORLD);
        std::sort(rootBuf, rootBuf+p*(p-1));
        
        for(int i=0;i<p-1;i++){
            broadCastArray[i]=rootBuf[(i+1)*p-1];
            printf("broadCastArray %d",broadCastArray[i]);
        }
        
    }
    
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
    
  // root process broadcasts splitters to all other processes
    
    
    MPI_Bcast(broadCastArray,p-1,MPI_INT,root,MPI_COMM_WORLD);
    for(int i=0;i<p-1;i++){
        printf("bcast: %d %d",rank, broadCastArray[i]);
    }
    */
    
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

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  // do a local sort of the received data

  // every process writes its result to a file

  free(vec);
  MPI_Finalize();
  return 0;
}
