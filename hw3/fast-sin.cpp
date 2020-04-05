#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"
#include <smmintrin.h>
// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
static constexpr double c2  = -1/((double)2);
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c12 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
static constexpr double halfpi=M_PI/2;
static constexpr double quarterpi=M_PI/4;
static constexpr double doublepi=M_PI*2;

// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}


void sin4_taylor_mod(double* sinx, const double* x){
    for(int i=0;i<4;i++){
        double x1=x[i];
        int temp=std::floor((x1+M_PI/4)/2/M_PI);
        x1=x1-temp*2*M_PI;  //x1 [-pi/4,7pi/4)
        int y=std::floor((x1+M_PI/4)*2/M_PI);
        x1=x1-M_PI*y/2;
        //int a=std::floor(y/2);
        //int b=y&1;
        //printf("%f ",x1);
        //double first=pow(-1,a)*b;
        //double second=pow(-1,a-1)*(b-1);
        int first=y/2;
        first=-first*2+1;
        //x1=(x1>3*M_PI/4)?x1-3*M_PI:x1;
        int second=y&1;
        //printf("%d %d %f\n",first,second,x1);
        
        
        double s=0;
        
        double x2=x1*x1;
        double x3=x1*x2;
        double x4=x2*x2;
        double x5=x2*x3;
        double x6=x3*x3;
        double x7=x3*x4;
        double x8=x4*x4;
        double x9=x4*x5;
        double x10=x5*x5;
        double x11=x5*x6;
        double x12=x6*x6;
        
        s += first  *second      *1.0;
        s += first  *(1-second)  *x1;
        s += first  *second      *x2  *c2;
        s += first  *(1-second)  *x3  *c3;
        s += first  *second      *x4  *c4;
        s += first  *(1-second)  *x5  *c5;
        s += first  *second      *x6  *c6;
        s += first  *(1-second)  *x7  *c7;
        s += first  *second      *x8  *c8;
        s += first  *(1-second)  *x9  *c9;
        s += first  *second      *x10 *c10;
        s += first  *(1-second)  *x11 *c11;
        s += first  *second      *x12 *c12;
        
        sinx[i]=s;
    }
}





void sin4_intrin_mod(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,first,second,third,temp,y;
    x1  = _mm256_load_pd(x);
  
  temp=_mm256_floor_pd(_mm256_div_pd(_mm256_add_pd(x1,_mm256_set1_pd(quarterpi)),_mm256_set1_pd(doublepi)));
  x1=_mm256_sub_pd(x1,_mm256_mul_pd(temp,_mm256_set1_pd(doublepi)));
    
  
  y=_mm256_floor_pd(_mm256_div_pd(_mm256_add_pd(x1,_mm256_set1_pd(quarterpi)),_mm256_set1_pd(halfpi)));
  x1=_mm256_sub_pd(x1,_mm256_mul_pd(y,_mm256_set1_pd(halfpi)));
  
  //first=_mm256_sub_pd(_mm256_set1_pd(1),_mm256_and_pd(y,_mm256_set1_pd(2)));
  //first=_mm256_sub_pd(_mm256_mul_pd(first,_mm256_set1_pd(2)),_mm256_set1_pd(1));
  //x1=(x1>3*M_PI/4)?x1-3*M_PI:x1;
  temp=_mm256_floor_pd(_mm256_div_pd(y,_mm256_set1_pd(2)));
  first=_mm256_sub_pd(_mm256_set1_pd(1),_mm256_mul_pd(temp,_mm256_set1_pd(2)));
    
  
  second=_mm256_sub_pd(y,_mm256_mul_pd(temp,_mm256_set1_pd(2)));
    
  third=_mm256_sub_pd(_mm256_set1_pd(1),second);
    
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x4  = _mm256_mul_pd(x2, x2);
  x5  = _mm256_mul_pd(x2, x3);
  x6  = _mm256_mul_pd(x3, x3);
  x7  = _mm256_mul_pd(x2, x5);
  x8  = _mm256_mul_pd(x4, x4);
  x9  = _mm256_mul_pd(x2, x7);
  x10  = _mm256_mul_pd(x5, x5);
  x11 = _mm256_mul_pd(x2, x9);
  x12  = _mm256_mul_pd(x6, x6);
    

  __m256d s =_mm256_set1_pd(0);
    
  s = _mm256_add_pd(s, _mm256_mul_pd(first,second));
  s = _mm256_add_pd(s, _mm256_mul_pd(x1 ,_mm256_mul_pd(first,third)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x2 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c2))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 ,_mm256_mul_pd(_mm256_mul_pd(first,third),_mm256_set1_pd(c3))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x4 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c4))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 ,_mm256_mul_pd(_mm256_mul_pd(first,third),_mm256_set1_pd(c5))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x6 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c6))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 ,_mm256_mul_pd(_mm256_mul_pd(first,third),_mm256_set1_pd(c7))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x8 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c8))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 ,_mm256_mul_pd(_mm256_mul_pd(first,third),_mm256_set1_pd(c9))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x10 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c10))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 ,_mm256_mul_pd(_mm256_mul_pd(first,third),_mm256_set1_pd(c11))));
  s = _mm256_add_pd(s, _mm256_mul_pd(x12 ,_mm256_mul_pd(_mm256_mul_pd(first,second),_mm256_set1_pd(c12))));
    
  _mm256_store_pd(sinx, s);
#else
  sin4_reference(sinx, x);
#endif
}




void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}

void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3,x5,x7,x9,x11;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x2, x3);
  x7  = _mm256_mul_pd(x2, x5);
  x9  = _mm256_mul_pd(x2, x7);
  x11 = _mm256_mul_pd(x2, x9);
    

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
    
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3,x5,x7,x9,x11;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5  = _mm_mul_pd(x3, x2);
    x7  = _mm_mul_pd(x5, x2);
    x9  = _mm_mul_pd(x7, x2);
    x11  = _mm_mul_pd(x9, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 )));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}



void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3,x5,x7,x9,x11;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;

  Vec4 s = x1;
  s += x3  * c3 + x5  * c5 + x7 * c7 + x9 * c9 + x11 * c11;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}


int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
    
    
    
    
  double* x_full = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref_full = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor_full = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin_full = (double*) aligned_malloc(N*sizeof(double));
  
  for (long i = 0; i < N; i++) {
    x_full[i] = (drand48()-0.5) *100* M_PI/2; // [-pi/4,pi/4]
    sinx_ref_full[i] = 0;
    sinx_taylor_full[i] = 0;
    sinx_intrin_full[i] = 0;
    
  }
  
    
  printf("Below is the testing for extra point.\n");
  tt.tic();
 for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref_full+i, x_full+i);
    }
 }
  printf("Reference time: %6.4f\n", tt.toc());
    
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor_mod(sinx_taylor_full+i, x_full+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref_full, sinx_taylor_full, N));
    
  
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin_mod(sinx_intrin_full+i, x_full+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref_full, sinx_intrin_full, N));
 
    
    
    aligned_free(x_full);
    aligned_free(sinx_ref_full);
    aligned_free(sinx_taylor_full);
    aligned_free(sinx_intrin_full);
}

