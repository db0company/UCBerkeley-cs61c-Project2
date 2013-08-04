#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <omp.h>

/* ************************************************************************* */
/* Tools and Constants                                                       */
/* ************************************************************************* */
//the unit of these sizes are "single precision floating point number(s)"
static int xmm_size = 4;
static int cache_size = 8192; //32KiB / 4 Bytes
static int cache_size_one_third = 2048; 
static int dimMax = 64; // sqrt(8182 spfpn / 2 matrices) ~ nearest mul of 4 
// static int dimMax = 52; // sqrt(8182 spfpn / 3 matrices) ~ nearest mul of 4 
static int block_size = 16;

/* Call me when *malloc fails */
void error(char * msg) {
  fprintf(stderr, "[Error] %s\n", msg);
  exit(EXIT_FAILURE);
}
void errorAllocation(void) {
  error("Allocation failure");
}
/* Tool for printing out __m128 types */
void printIntel(__m128 var){
  printf("{%f, %f, %f, %f}\n", 
    *((float*)(&var)+0),
    *((float*)(&var)+1),
    *((float*)(&var)+2),
    *((float*)(&var)+3));
}

/* ************************************************************************* */
/* SGEMM                                                                     */
/* ************************************************************************* */

#define PTHREAD_THREADS_MAX 16

typedef struct {
  ssize_t thread_number;
  int n; // will not change
  int m; // will not change
  float * A;
  float * B;
  float * C;
} threadParam;

void * threaded_sgemm(void * vp) {
  threadParam * p = vp;

  for (int C_K = 0; C_K < p->n; C_K += dimMax) {
    
    for (int C_I = 0; C_I < p-> m; C_I += dimMax) {

      __m128 c1;
      __m128 c2;
      __m128 c3;
      __m128 c4;
      __m128 a1;
      __m128 a2;
      __m128 a3;
      __m128 a4;
      int minK;
      int minI;
      int minJ;

      int chunk_size = p->m / PTHREAD_THREADS_MAX;
      for (int C_J = p->thread_number * chunk_size;
	   C_J < ((p->thread_number + 1) * chunk_size);
	   C_J += dimMax) { // todo: not sure if it works with a gap of more than 1

	minK = (C_K + dimMax < p->n) ? (C_K + dimMax) : p->n;
	minI = (C_I + dimMax < p->m) ? (C_I + dimMax) : p->m;
	minJ = (C_J + dimMax < p->m) ? (C_J + dimMax) : p->m;

	for (int j = C_J; j < minJ; j += 4){
	  for (int i = C_I; i < minI; i += 4){
	    //Obtain placeholders for summation
	    c1 = _mm_setzero_ps();
	    c2 = _mm_setzero_ps();
	    c3 = _mm_setzero_ps();
	    c4 = _mm_setzero_ps();

	    for (int k = C_K; k < minK; k += 4) {
	      //Obtain grouped single precision of 4 from columns a1, a2, a3, a4
	      a1 = _mm_loadu_ps(p->A + i + p->m * (k + 0));
	      a2 = _mm_loadu_ps(p->A + i + p->m * (k + 1));
	      a3 = _mm_loadu_ps(p->A + i + p->m * (k + 2));
	      a4 = _mm_loadu_ps(p->A + i + p->m * (k + 3));

	      //To see the full formula on how positions of entries of B are calculated, refer to
	      //bottom of this file
	      /*b1x x a1, for formula explanation, refer to bottom of this file*/
	      c1 = _mm_add_ps(c1, _mm_mul_ps(a1, _mm_load1_ps((p->B + j + 0 + p->m * (k + 0)))));
	      c2 = _mm_add_ps(c2, _mm_mul_ps(a1, _mm_load1_ps((p->B + j + 1 + p->m * (k + 0)))));
	      c3 = _mm_add_ps(c3, _mm_mul_ps(a1, _mm_load1_ps((p->B + j + 2 + p->m * (k + 0)))));
	      c4 = _mm_add_ps(c4, _mm_mul_ps(a1, _mm_load1_ps((p->B + j + 3 + p->m * (k + 0)))));

	      // b2x x a2, for formula explanation, refer to bottom of this file
	      c1 = _mm_add_ps(c1, _mm_mul_ps(a2, _mm_load1_ps((p->B + j + 0 + p->m * (k + 1)))));
	      c2 = _mm_add_ps(c2, _mm_mul_ps(a2, _mm_load1_ps((p->B + j + 1 + p->m * (k + 1)))));
	      c3 = _mm_add_ps(c3, _mm_mul_ps(a2, _mm_load1_ps((p->B + j + 2 + p->m * (k + 1)))));
	      c4 = _mm_add_ps(c4, _mm_mul_ps(a2, _mm_load1_ps((p->B + j + 3 + p->m * (k + 1)))));

	      /*b3x x a3, for formula explanation, refer to bottom of this file*/
	      c1 = _mm_add_ps(c1, _mm_mul_ps(a3, _mm_load1_ps((p->B + j + 0 + p->m * (k + 2)))));
	      c2 = _mm_add_ps(c2, _mm_mul_ps(a3, _mm_load1_ps((p->B + j + 1 + p->m * (k + 2)))));
	      c3 = _mm_add_ps(c3, _mm_mul_ps(a3, _mm_load1_ps((p->B + j + 2 + p->m * (k + 2)))));
	      c4 = _mm_add_ps(c4, _mm_mul_ps(a3, _mm_load1_ps((p->B + j + 3 + p->m * (k + 2)))));

	      /*b4x x a4, for formula explanation, refer to bottom of this file*/
	      c1 = _mm_add_ps(c1, _mm_mul_ps(a4, _mm_load1_ps((p->B + j + 0 + p->m * (k + 3)))));
	      c2 = _mm_add_ps(c2, _mm_mul_ps(a4, _mm_load1_ps((p->B + j + 1 + p->m * (k + 3)))));
	      c3 = _mm_add_ps(c3, _mm_mul_ps(a4, _mm_load1_ps((p->B + j + 2 + p->m * (k + 3)))));
	      c4 = _mm_add_ps(c4, _mm_mul_ps(a4, _mm_load1_ps((p->B + j + 3 + p->m * (k + 3)))));
	    }
	      
	    //Accumulate sum in C
	    // todo: critical?
	    _mm_storeu_ps(p->C + i + p->m * (j + 0), 
			  _mm_add_ps(c1, _mm_loadu_ps(p->C + i + p->m * (j + 0))));
	    _mm_storeu_ps(p->C + i + p->m * (j + 1), 
			  _mm_add_ps(c2, _mm_loadu_ps(p->C + i + p->m * (j + 1))));
	    _mm_storeu_ps(p->C + i + p->m * (j + 2), 
			  _mm_add_ps(c3, _mm_loadu_ps(p->C + i + p->m * (j + 2))));
	    _mm_storeu_ps(p->C + i + p->m * (j + 3), 
			  _mm_add_ps(c4, _mm_loadu_ps(p->C + i + p->m * (j + 3))));
	  }
	}
      }
    }
  }
  return NULL;
}

void sgemm(int m, int n, float * A, float * B, float * C) {
  float* original_C = C;
  bool padded = false;
  int old_m;
  int old_n;

  //Pad the matrix with zeros if necessary
  if (m % dimMax || n % xmm_size){

    padded = true;
    old_m = m;
    old_n = n;

    //Padded sizes are to the nearest multiple of xmm_size
    int padded_m = ((m -1) / dimMax +1) * dimMax;
    int padded_n = ((n -1) / xmm_size +1) * xmm_size;

    //New Matrix A, column major
    float *A_padded = (float*)calloc(padded_m*padded_n, sizeof(float));
    if (!A_padded) errorAllocation();
    //New Matrix B, row major
    float *B_padded = (float*)calloc(padded_m*padded_n, sizeof(float));
    if (!B_padded) errorAllocation();

    //Transfer data over from old matrices
    for (int n_i = 0; n_i < n; ++n_i){
      memcpy(A_padded+n_i*padded_m, A+n_i*old_m, old_m*sizeof(float));
      memcpy(B_padded+n_i*padded_m, B+n_i*old_m, old_m*sizeof(float));
    }

    //Resize matrix C, data not important
    float *C_padded = (float*)calloc(padded_m*padded_m, sizeof(float));
    if (!C_padded) errorAllocation();

    //reflect changes
    A = A_padded;
    B = B_padded;
    C = C_padded;
    m = padded_m;
    n = padded_n;
  }

  // thread the main execution
  pthread_t t[PTHREAD_THREADS_MAX + 1];
  threadParam p; // contain all the shared variables
  p.n = n;
  p.m = m;
  p.A = A;
  p.B = B;
  p.C = C;
  for (ssize_t thread_num = 0; thread_num < PTHREAD_THREADS_MAX; ++thread_num) {
    threadParam newp = p;
    newp.thread_number = thread_num;
    if ((pthread_create(&t[thread_num], NULL, threaded_sgemm, &newp)) != 0)
      error("Pthread not created");
  }

  // wait for all the threads to terminate
  for (ssize_t thread_num = 0; thread_num < PTHREAD_THREADS_MAX; ++thread_num) {
    void * n;
    pthread_join(t[thread_num], &n);
  }
    
  //Un-pad the matrix by transferring useful data back to original C's memory location
  if (padded){
    for (int n_i = 0; n_i < old_m; ++n_i){
      memcpy(original_C+n_i*old_m, C+n_i*m, old_m*sizeof(float));
    }
 }
}

/* ************************************************************************* */
/* Documentation

          R e g i s t e r   B l o c k i n g 
          Break the matrices into blocks of 4x4 to utilize 12 xmm registers
          
          D i a g r a m
          a1, a2, a3, and a4 refer to columns of matrix A
          b1x, b2x, b3x, and b4x refer to entries in each row of matrix B    
          c1, c2, c3, and c4 refer to columns matrix C

          size: m (height) x n (width)
           C[i+j*m] += A[i+k*m] x B[k*m+j]

             k*m                j                  j*m
           +------+          +------+            +------+
           |+     |          |+---->|            |+     |
         i || A   |    X  k*m|  B   |      =   i || C   |
           |v     |          |      |            |v     |
           +------+          +------+            +------+

            col-maj          row-maj             col-maj


            i = j = m
            k = n
            A and B are guaranteed to be mxn & nxm
            +---> means flow direction of array, which dictates k*m = calculation
            of columns for A and rows for B

/* ************************************************************************* */
/* Code Bank and References                                                  */
/* ************************************************************************* */

/*
        Positions for matrix B

        // float* b11 = (B + j+0 + m_a*(k+0));
        // float* b12 = (B + j+1 + m_a*(k+0));
        // float* b13 = (B + j+2 + m_a*(k+0));
        // float* b14 = (B + j+3 + m_a*(k+0));

        // float* b21 = (B + j+0 + m_a*(k+1));
        // float* b22 = (B + j+1 + m_a*(k+1));
        // float* b23 = (B + j+2 + m_a*(k+1));
        // float* b24 = (B + j+3 + m_a*(k+1));

        // float* b31 = (B + j+0 + m_a*(k+2));
        // float* b32 = (B + j+1 + m_a*(k+2));
        // float* b33 = (B + j+2 + m_a*(k+2));
        // float* b34 = (B + j+3 + m_a*(k+2));

        // float* b41 = (B + j+0 + m_a*(k+3));
        // float* b42 = (B + j+1 + m_a*(k+3));
        // float* b43 = (B + j+2 + m_a*(k+3));
        // float* b44 = (B + j+3 + m_a*(k+3));

        Math formula for matrix multiplication with register blocking
        
        for matrix A_sub 4x4 multiply matrix B_sub 4x4, 
        it is equivalent to each B's entry duplicated into a column vector in R^4
        and the dot product of that column vector and columns of A

        /*b1 x a1*/
        // c1 += 4{b11} dot a1
        // c2 += 4{b12} dot a1
        // c3 += 4{b13} dot a1
        // c4 += 4{b14} dot a1
        /*b2 x a2*/
        // c1 += 4{b21} dot a2
        // c2 += 4{b22} dot a2
        // c3 += 4{b23} dot a2
        // c4 += 4{b24} dot a2
        /*b3 x a3*/
        // c1 += 4{b31} dot a3
        // c2 += 4{b32} dot a3
        // c3 += 4{b33} dot a3
        // c4 += 4{b34} dot a3
        /*b4 x a4*/
        // c1 += 4{b41} dot a4
        // c2 += 4{b42} dot a4
        // c3 += 4{b43} dot a4
        // c4 += 4{b44} dot a4


