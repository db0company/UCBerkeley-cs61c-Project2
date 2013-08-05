#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>


/* ************************************************************************* */
/* Tools and Constants                                                       */
/* ************************************************************************* */
//the unit of these sizes are "single precision floating point number(s)"
// static int xmm_size = 4;
// static int cache_size = 8192; //32KiB / 4 Bytes
// static int cache_size_one_third = 2048; 
static int dimMax = 16; // sqrt(8182 spfpn / 2 matrices) ~ nearest mul of 4 
// static int dimMax = 52; // sqrt(8182 spfpn / 3 matrices) ~ nearest mul of 4 
// static int block_size = 16;

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


void sgemm(int m, int n, float * A, float * B, float * C) {

/*
Specs: 
cache blocking by dealing with 52x52 matrices each time 
loop ordering by k-j-i or k-i-j, K outermost
false sharing prevention: parallelize 
*/
  if (n % dimMax){
    int old_n = n;
    n = ((n -1) / dimMax +1) * dimMax;

    #pragma omp parallel sections
    {
      #pragma omp section
      {
        //New Matrix A, column major
        float *A_padded = (float*)calloc(m*n, sizeof(float));
        if (!A_padded) errorAllocation();
        memcpy(A_padded, A, m*old_n*sizeof(float));
        A = A_padded;
      }
      #pragma omp section
      {
        //New Matrix B, row major
        float *B_padded = (float*)calloc(m*n, sizeof(float));
        if (!B_padded) errorAllocation();
        memcpy(B_padded, B, m*old_n*sizeof(float));
        B = B_padded;
      }
    }
    
    
  }
  // __m128 c1;
  // __m128 c2;
  // __m128 c3;
  // __m128 c4;
  // __m128 a1;
  // __m128 a2;
  // __m128 a3;
  // __m128 a4;
  // __m128 b1x;
  // __m128 b2x;
  // __m128 b3x;
  // __m128 b4x;

  #pragma omp parallel
  {
  // #pragma omp for private(c1, c2, c3, c4, a1, a2, a3, a4, b1x, b2x, b3x, b4x) firstprivate(A, B, C, m, n)  
  for (int C_I = 0; C_I < m/dimMax*dimMax; C_I+=dimMax){
    // #pragma omp for private(c1, c2, c3, c4, a1, a2, a3, a4, b1x, b2x, b3x, b4x) firstprivate(A, B, C, m, n)  
    for (int C_K = 0; C_K < n/dimMax*dimMax; C_K+=dimMax){
      #pragma omp for firstprivate(A, B, C, m, n)  
      for (int C_J = 0; C_J < m/dimMax*dimMax; C_J+=dimMax){

        for (int j = C_J; j < C_J+dimMax; j+=4){
          for (int k = C_K; k < C_K+dimMax; k+=4){
            for (int i = C_I; i < C_I+dimMax; i+=4){

              __m128 c1 = _mm_setzero_ps();
              __m128 c2 = _mm_setzero_ps();
              __m128 c3 = _mm_setzero_ps();
              __m128 c4 = _mm_setzero_ps();

              __m128 a1 = _mm_loadu_ps(A + i + m*(k+0));
              __m128 a2 = _mm_loadu_ps(A + i + m*(k+1));
              __m128 a3 = _mm_loadu_ps(A + i + m*(k+2));
              __m128 a4 = _mm_loadu_ps(A + i + m*(k+3));

              __m128 b1x = _mm_loadu_ps(B + j+0 + m*(k+0));
              __m128 b2x = _mm_loadu_ps(B + j+0 + m*(k+1));;
              __m128 b3x = _mm_loadu_ps(B + j+0 + m*(k+2));;
              __m128 b4x = _mm_loadu_ps(B + j+0 + m*(k+3));;

              // #pragma omp sections nowait
              {
              /*b1x x a1, for formula explanation, refer to bottom of this file*/
              // #pragma omp section
              {
              c1 = _mm_add_ps(c1, _mm_mul_ps(a1, _mm_shuffle_ps(b1x, b1x, 0x00)));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a1, _mm_shuffle_ps(b1x, b1x, 0x55)));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a1, _mm_shuffle_ps(b1x, b1x, 0xAA)));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a1, _mm_shuffle_ps(b1x, b1x, 0xFF)));
              }
              // #pragma omp section
              {
              // b2x x a2, for formula explanation, refer to bottom of this file
              c1 = _mm_add_ps(c1, _mm_mul_ps(a2, _mm_shuffle_ps(b2x, b2x, 0x00)));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a2, _mm_shuffle_ps(b2x, b2x, 0x55)));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a2, _mm_shuffle_ps(b2x, b2x, 0xAA)));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a2, _mm_shuffle_ps(b2x, b2x, 0xFF)));
              }
              // #pragma omp section
              {
              /*b3x x a3, for formula explanation, refer to bottom of this file*/
              c1 = _mm_add_ps(c1, _mm_mul_ps(a3, _mm_shuffle_ps(b3x, b3x, 0x00)));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a3, _mm_shuffle_ps(b3x, b3x, 0x55)));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a3, _mm_shuffle_ps(b3x, b3x, 0xAA)));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a3, _mm_shuffle_ps(b3x, b3x, 0xFF)));
              }
              // #pragma omp section
              {
              /*b4x x a4, for formula explanation, refer to bottom of this file*/
              c1 = _mm_add_ps(c1, _mm_mul_ps(a4, _mm_shuffle_ps(b4x, b4x, 0x00)));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a4, _mm_shuffle_ps(b4x, b4x, 0x55)));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a4, _mm_shuffle_ps(b4x, b4x, 0xAA)));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a4, _mm_shuffle_ps(b4x, b4x, 0xFF)));
              }
              }
            _mm_storeu_ps(C + i + m*(j+0), _mm_add_ps(c1, _mm_loadu_ps(C + i + m*(j+0))));
            _mm_storeu_ps(C + i + m*(j+1), _mm_add_ps(c2, _mm_loadu_ps(C + i + m*(j+1))));
            _mm_storeu_ps(C + i + m*(j+2), _mm_add_ps(c3, _mm_loadu_ps(C + i + m*(j+2))));
            _mm_storeu_ps(C + i + m*(j+3), _mm_add_ps(c4, _mm_loadu_ps(C + i + m*(j+3))));
          
            }

            //Accumulate sum in C
            
  
          }
        }
      }
    }
  }//end for
  }//end parallel


  // if (m%dimMax || n%dimMax){
  if (m%dimMax){
  // if (false){
    #pragma omp parallel
    {
    int fringe_m = m / dimMax * dimMax;
    int fringe_n = n / dimMax * dimMax;
    int i;
    int k;
    int j;

    if (m%dimMax){
      #pragma omp for nowait
      for (j = fringe_m; j < m; j++){
        for (k = 0; k < n; k++){
          for (i = 0; i < fringe_m; i++){
            C[i+j*m] += A[i+k*m] * B[j+k*m];
          }
        }
      }
      #pragma omp for nowait
      for (j = 0; j < m; j++){
        for (k = 0; k < n; k++){
          for (i = fringe_m; i < m; i++){
            C[i+j*m] += A[i+k*m] * B[j+k*m];
          }
        }
      }
    }

    // if (n%dimMax){
    //   #pragma omp for nowait
    //   // int chunk_size = fringe_m/omp_get_num_threads();
    //   // int limit = chunk_size*(omp_get_thread_num()+1);
    //   // j = limit - chunk_size;
    //   // if (limit+chunk_size>fringe_m) limit = fringe_m;
    //   // printf("Thread %d has limit: %d and chunk_size: %d\n",omp_get_thread_num(), limit, chunk_size);
    //   // for (; j < limit; j++){
    //   for (j = 0; j < fringe_m; j++){
    //     for (k = fringe_n; k < n; k++){
    //       for (i = 0; i < fringe_m; i++){
    //         C[i+j*m] += A[i+k*m] * B[j+k*m];
    //       }
    //     }
    //   }
    // }
    }//end parallel
    // printf("================================\n"); 
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


