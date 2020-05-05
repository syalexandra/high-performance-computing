/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;
    
  for(i=1;i<=lN;i++){
      for (j = 1; j <= lN; j++){
          tmp = ((4.0*lu[i*(lN+2)+j] - lu[(i-1)*(lN+2)+j] - lu[(i+1)*(lN+2)+j-1] - lu[i*(lN+2)+j+1])* invhsq - 1);
          lres += tmp * tmp;
      }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, j,p, Nl, N,lN, iter, max_iters;
  MPI_Status status, status1,status2,status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);
    
    //printf("Nl: \n");
  sscanf(argv[1], "%d", &Nl);
    //printf("max_iteration: \n");
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
    int ptemp=p;
    j=0;
    while(ptemp>0 && ptemp!=1){
        if(ptemp % 4 != 0){
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        else{
            ptemp = ptemp/4;
            j+=1;
        }
    }
    
    N=(1<<j)*Nl;
    
    
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();
    
  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc((Nl + 2)*(Nl + 2),sizeof(double));
  double * lunew = (double *) calloc((Nl+ 2)*(Nl+ 2),sizeof(double));
  double * lutemp;
    
    double * leftin = (double *) calloc(Nl,sizeof(double));
     double * rightin = (double *) calloc(N1,sizeof(double));
     double * leftout = (double *) calloc(N1,sizeof(double));
     double * rightout = (double *) calloc(N1,sizeof(double));

    
    
  double h = 1.0 / (Nl + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, Nl, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (int i = 1; i <= Nl; i++){
        for(int k=1;k<=Nl;k++){
            lunew[i*(Nl+2)+k]  = 0.25 * (hsq + lu[(i - 1)*(Nl+2)+k] + lu[(i + 1)*(Nl+2)+k]+lu[i*(Nl+2)+k-1]+lu[i*(Nl+2)+(k+1)]);
        }
    }
      gres =compute_residual(lunew, Nl, invhsq);
      //printf("%f \n",gres);
      
      int psqrt=sqrt(p);
      int mpirankX=mpirank /psqrt;
      int mpirankY=mpirank % psqrt;
      

    /* communicate ghost values */
    if (mpirankX < psqrt - 1) {
      /* If not the last process, send/recv bdry values to the right */
      MPI_Send(&(lunew[Nl*(Nl+2)+1]), Nl, MPI_DOUBLE, mpirank+psqrt, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(Nl+1)*(Nl+2)+1]), Nl, MPI_DOUBLE, mpirank+psqrt, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirankX > 0) {
      /* If not the first process, send/recv bdry values to the left */
      MPI_Send(&(lunew[1+(Nl+2)*1]), Nl, MPI_DOUBLE, mpirank-psqrt, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]), Nl, MPI_DOUBLE, mpirank-psqrt, 124, MPI_COMM_WORLD, &status1);
    }
      
      printf("rank %d ",mpirank);
    for(int i=0;i<Nl;i++){
        leftout[i]=lunew[(i+1)*(Nl+2)+1];
        rightout[i]=lunew[(i+1)*(Nl+2)+Nl];
        printf("%f ",leftout[i]);
    }
      printf("\n");
       
    if(mpirankY<psqrt-1){
        MPI_Send(&(rightout[0]), Nl, MPI_DOUBLE, mpirank+1, 125, MPI_COMM_WORLD);
        MPI_Recv(&(rightin[0]), Nl, MPI_DOUBLE, mpirank+1, 126, MPI_COMM_WORLD, &status2);
        for(int i=0;i<Nl;i++){
            lunew[(i+1)*(Nl+2)+Nl+1]=rightin[i];
        }
    }
      
      

    if(mpirankY>0){
        MPI_Send(&(leftout[0]), Nl, MPI_DOUBLE, mpirank-1, 126, MPI_COMM_WORLD);
        MPI_Recv(&(leftin[0]), Nl, MPI_DOUBLE, mpirank-1, 125, MPI_COMM_WORLD, &status3);
        for(int i=0;i<Nl;i++){
            lunew[(i+1)*(Nl+2)]=leftin[i];
        }
    }
      
      
      
    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    //if (0 == (iter % 10)) {
      gres = compute_residual(lu, Nl, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    //}
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
