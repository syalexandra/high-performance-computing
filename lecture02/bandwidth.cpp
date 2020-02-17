// Timing memory operations and calculating bandwidth
// $ g++ -std=c++11 -O3 -march=native bandwidth.cpp && ./a.out -n 400000000 -repeat 1 -skip 1
// $ cat /proc/cpuinfo
// $ cat /proc/meminfo
// $ htop
// $ getconf -a | grep CACHE
// ADVANCED $ valgrind --tool=cachegrind ./a.out

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  long repeat = read_option<long>("-repeat", argc, argv, "1");
  long skip = read_option<long>("-skip", argc, argv, "1");
  
  t.tic();
  double* x = (double*) malloc(n * sizeof(double)); // dynamic allocation on heap
  //double x[400000000]; // static allocation on stack
  printf("time to malloc         = %f s\n", t.toc());

  t.tic();
  for (long i = 0; i < n; i += skip) x[i] = i;
  printf("time to initialize     = %f s\n", t.toc());

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = k;
    //#pragma omp parallel for schedule (static)
    for (long i = 0; i < n; i += skip) {
      x[i] = kk;
    }
  }
  printf("time to write          = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = x[0] * 0.5;
    //#pragma omp parallel for schedule (static)
    for (long i = 0; i < n; i += skip) {
      x[i] = x[i] * 0.5 + kk;
    }
  }
  printf("time to read + write   = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", 2 * n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  double sum = 0;
  for (long k = 0; k < repeat; k++) {
    //#pragma omp parallel for schedule (static) reduction(+:sum)
    for (long i = 0; i < n; i += skip) {
      sum += x[i];
    }
  }
  printf("time to read           = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  //free(x);
  printf("time to free           = %f s\n", t.toc());

  return sum;
}

// Synopsis
//
// Memory Allocation: malloc return a virtual memory address such that the
// requested memory range of memory addresses don't overlap with another
// allocation. This is managed with a heap data structure and can take a
// non-negligible amount of time.  By contrast, allocations on the stack are
// instantaneous and the virtual memory address is just the address of the top
// of the stack. C++ (unlike C and FORTRAN) only allows fixed size (or static)
// allocations on the stack i.e. the array size must be known at compile time.
// ** Avoid frequent allocations on the heap for small sized arrays, use static
// allocation on the stack if possible **
//
// Freeing Memory: This involves returning the freed memory to the heap and
// freeing the corresponding memory pages. The latter is handled by the OS
// kernel and had non-negligible cost.  By contrast, freeing memory from the
// stack is instantaneous.
//
// Memory Initialization: When a new array is accessed for the first time, the
// processor looks for a translation of the virtual memory address to the
// physical memory address (acrual address on the RAM) in the Translation
// Lookaside Buffer (TLB). Since the mapping does not yet exist, it results in
// a page fault. This causes a context switch to the OS kernel which then adds
// an entry in the page table. The context switches back to the user
// application and after the address translation, the processor tries to load
// the data from the corresponding physical address. Memeory pages are
// generally of size 4KB (but can also be larger 2MB or 1GB -- see hugepages).
// ** Page faults can be expensive; therefore, it sometimes makes sense to use
// preallocated memory buffers instead of allocating new memory on the heap
// each time **
//
// Reading from Memory:
// Try to find the cache line (generally of size 64B) first in the L1 cache.
// If not found in L1, it then looks for it in L2 cache and so on.  If the
// cache line is not found, then the virtual memory address is translated to a
// physical memory address using TLB lookup and the cache line is loaded from
// the main memory (RAM) and brought to the L3 cache, the L2 cache and then to
// the L1 cache. The data is finally read into a register from the L1 cache.
// ** Caches lower in the memory hierarchy have higher latency and lower
// bandwidth. So, try to keep data closer to the processor in registers, then
// L1 cache, L2 cache, L3 cache **
// ** To readuce the overhead (latency) of cache misses, there are special
// instructions for prefetching which allow us to initiate the process of
// loading of a cache line several clock cycles in advance. These special
// instructions will be discussed in future lectures **
//
// Writing to Memory: The system translates the virtual address to the physical
// address and then looks for the corresponding cache line in L1; if not found,
// it is loaded from lower level caches or main memory. Then, the cache line in
// L1 is then updated with new data and the corresonding cache lines are marked
// dirty in L2 and L3 cache. The the update to the cache line in L1 propagates
// to the L2 cache when either the cache line is explicitly flushed by using a
// special instruction or flushed implicitly when that cache line must be
// replaced by new data breought in to the L1 cache. Similar propagation
// happens L2 to L3 and from L3 to main memory.
// ** Note that writing data also requires first reading the corresponding
// cache line, this can be wasteful if the entire cache line needs to be
// overwritten anyway. This can be mitigated using special instructions for
// non-temporal writes which don't load the cache line before writing and write
// directly to the main memory (not completely sure about that). These special
// instructions will be discussed in later lectures **
//
