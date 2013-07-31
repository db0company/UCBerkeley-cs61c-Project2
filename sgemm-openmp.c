#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <omp.h>


/* ************************************************************************* */
/* Tools and Constants                                                       */
/* ************************************************************************* */
//the unit of these sizes are "single precision floating point number(s)"
static int xmm_size = 4;
static int cache_size = 8192; //32KiB / 4 Bytes
static int cache_size_one_third = 2048; 
static int dimMax = 52; // sqrt(8182 spfpn / 3 matrices) ~ nearest mul of 4 
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


void sgemm(int m, int n, float * A, float * B, float * C) {
  float* original_C = C;
  bool padded = false;
  int old_m;
  int old_n;

  //Pad the matrix with zeros if necessary
  if (m % xmm_size || n % xmm_size){

    padded = true;
    old_m = m;
    old_n = n;

    //Padded sizes are to the nearest multiple of xmm_size
    int padded_m = ((m -1) / xmm_size +1) * xmm_size;
    int padded_n = ((n -1) / xmm_size +1) * xmm_size;

    //New Matrix A, column major
    float *A_padded = (float*)calloc(padded_m*padded_n, sizeof(float));
    if (!A_padded) errorAllocation();
    //New Matrix B, row major
    float *B_padded = (float*)calloc(padded_m*padded_n, sizeof(float));
    if (!B_padded) errorAllocation();

    //Transfer data over from old matrices
    // #pragma omp parallel for
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

  // for (int k = 0; k < n; k+=4){
  //     for (int j = 0; j < m; j+=4){
  //       for (int i = 0; i < m; i+=4){
  //         //pre-calculate the addresses for columns of C
  //         float* c1_addr = C + i + m*(j+0);
  //         float* c2_addr = C + i + m*(j+1);
  //         float* c3_addr = C + i + m*(j+2);
  //         float* c4_addr = C + i + m*(j+3);

  //         //Obtain placeholders for summation
  //         __m128 c1 = _mm_setzero_ps();
  //         __m128 c2 = _mm_setzero_ps();
  //         __m128 c3 = _mm_setzero_ps();
  //         __m128 c4 = _mm_setzero_ps();

  //         //Obtain grouped single precision of 4 from columns a1, a2, a3, a4
  //         __m128 a1 = _mm_loadu_ps(A + i + m*(k+0));
  //         __m128 a2 = _mm_loadu_ps(A + i + m*(k+1));
  //         __m128 a3 = _mm_loadu_ps(A + i + m*(k+2));
  //         __m128 a4 = _mm_loadu_ps(A + i + m*(k+3));

  //         //To see the full formula on how positions of entries of B are calculated, refer to
  //         //bottom of this file

  //         /*b1x x a1, for formula explanation, refer to bottom of this file*/
  //         c1 = _mm_add_ps(c1, _mm_mul_ps(a1, _mm_load1_ps((B + j+0 + m*(k+0)))));
  //         c2 = _mm_add_ps(c2, _mm_mul_ps(a1, _mm_load1_ps((B + j+1 + m*(k+0)))));
  //         c3 = _mm_add_ps(c3, _mm_mul_ps(a1, _mm_load1_ps((B + j+2 + m*(k+0)))));
  //         c4 = _mm_add_ps(c4, _mm_mul_ps(a1, _mm_load1_ps((B + j+3 + m*(k+0)))));

  //         /*b2x x a2, for formula explanation, refer to bottom of this file*/
  //         c1 = _mm_add_ps(c1, _mm_mul_ps(a2, _mm_load1_ps((B + j+0 + m*(k+1)))));
  //         c2 = _mm_add_ps(c2, _mm_mul_ps(a2, _mm_load1_ps((B + j+1 + m*(k+1)))));
  //         c3 = _mm_add_ps(c3, _mm_mul_ps(a2, _mm_load1_ps((B + j+2 + m*(k+1)))));
  //         c4 = _mm_add_ps(c4, _mm_mul_ps(a2, _mm_load1_ps((B + j+3 + m*(k+1)))));

  //         /*b3x x a3, for formula explanation, refer to bottom of this file*/
  //         c1 = _mm_add_ps(c1, _mm_mul_ps(a3, _mm_load1_ps((B + j+0 + m*(k+2)))));
  //         c2 = _mm_add_ps(c2, _mm_mul_ps(a3, _mm_load1_ps((B + j+1 + m*(k+2)))));
  //         c3 = _mm_add_ps(c3, _mm_mul_ps(a3, _mm_load1_ps((B + j+2 + m*(k+2)))));
  //         c4 = _mm_add_ps(c4, _mm_mul_ps(a3, _mm_load1_ps((B + j+3 + m*(k+2)))));

  //         /*b4x x a4, for formula explanation, refer to bottom of this file*/
  //         c1 = _mm_add_ps(c1, _mm_mul_ps(a4, _mm_load1_ps((B + j+0 + m*(k+3)))));
  //         c2 = _mm_add_ps(c2, _mm_mul_ps(a4, _mm_load1_ps((B + j+1 + m*(k+3)))));
  //         c3 = _mm_add_ps(c3, _mm_mul_ps(a4, _mm_load1_ps((B + j+2 + m*(k+3)))));
  //         c4 = _mm_add_ps(c4, _mm_mul_ps(a4, _mm_load1_ps((B + j+3 + m*(k+3)))));

  //         //Accumulate sum in C
  //         _mm_storeu_ps(c1_addr, _mm_add_ps(c1, _mm_loadu_ps(c1_addr)));
  //         _mm_storeu_ps(c2_addr, _mm_add_ps(c2, _mm_loadu_ps(c2_addr)));
  //         _mm_storeu_ps(c3_addr, _mm_add_ps(c3, _mm_loadu_ps(c3_addr)));
  //         _mm_storeu_ps(c4_addr, _mm_add_ps(c4, _mm_loadu_ps(c4_addr)));
  //       }
  //     }
  //   }

  // for( int k = 0; k < n; k++ ){
  //   for ( int i = 0; i < m; i++ ){
  //     for( int j = 0; j < m; j++ ) {
  //       C[i+j*m] += A[i+k*m] * B[j+k*m];
  //     }
  //   }
  // }

/*
Specs: 
cache blocking by dealing with 52x52 matrices each time 
loop ordering by k-j-i or k-i-j, K outermost
false sharing prevention: parallelize 
*/

  // #pragma omp parallel for
  for (int C_K = 0; C_K < n; C_K+=dimMax){
    for (int C_I = 0; C_I < m; C_I+=dimMax){
      for (int C_J = 0; C_J < m; C_J+=dimMax){

        int minK = (C_K+dimMax < n) ? (C_K+dimMax) : n;
        int minI = (C_I+dimMax < m) ? (C_I+dimMax) : m;
        int minJ = (C_J+dimMax < m) ? (C_J+dimMax) : m;

        
        for (int k = C_K; k < minK; k+=4){
          for (int i = C_I; i < minI; i+=4){
            for (int j = C_J; j < minJ; j+=4){

              float* c1_addr = C + i + m*(j+0);
              float* c2_addr = C + i + m*(j+1);
              float* c3_addr = C + i + m*(j+2);
              float* c4_addr = C + i + m*(j+3);

              //Obtain placeholders for summation
              __m128 c1 = _mm_setzero_ps();
              __m128 c2 = _mm_setzero_ps();
              __m128 c3 = _mm_setzero_ps();
              __m128 c4 = _mm_setzero_ps();

              //Obtain grouped single precision of 4 from columns a1, a2, a3, a4
              __m128 a1 = _mm_loadu_ps(A + i + m*(k+0));
              __m128 a2 = _mm_loadu_ps(A + i + m*(k+1));
              __m128 a3 = _mm_loadu_ps(A + i + m*(k+2));
              __m128 a4 = _mm_loadu_ps(A + i + m*(k+3));

              //To see the full formula on how positions of entries of B are calculated, refer to
              //bottom of this file

              /*b1x x a1, for formula explanation, refer to bottom of this file*/
              c1 = _mm_add_ps(c1, _mm_mul_ps(a1, _mm_load1_ps((B + j+0 + m*(k+0)))));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a1, _mm_load1_ps((B + j+1 + m*(k+0)))));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a1, _mm_load1_ps((B + j+2 + m*(k+0)))));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a1, _mm_load1_ps((B + j+3 + m*(k+0)))));

              // b2x x a2, for formula explanation, refer to bottom of this file
              c1 = _mm_add_ps(c1, _mm_mul_ps(a2, _mm_load1_ps((B + j+0 + m*(k+1)))));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a2, _mm_load1_ps((B + j+1 + m*(k+1)))));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a2, _mm_load1_ps((B + j+2 + m*(k+1)))));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a2, _mm_load1_ps((B + j+3 + m*(k+1)))));

              /*b3x x a3, for formula explanation, refer to bottom of this file*/
              c1 = _mm_add_ps(c1, _mm_mul_ps(a3, _mm_load1_ps((B + j+0 + m*(k+2)))));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a3, _mm_load1_ps((B + j+1 + m*(k+2)))));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a3, _mm_load1_ps((B + j+2 + m*(k+2)))));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a3, _mm_load1_ps((B + j+3 + m*(k+2)))));

              /*b4x x a4, for formula explanation, refer to bottom of this file*/
              c1 = _mm_add_ps(c1, _mm_mul_ps(a4, _mm_load1_ps((B + j+0 + m*(k+3)))));
              c2 = _mm_add_ps(c2, _mm_mul_ps(a4, _mm_load1_ps((B + j+1 + m*(k+3)))));
              c3 = _mm_add_ps(c3, _mm_mul_ps(a4, _mm_load1_ps((B + j+2 + m*(k+3)))));
              c4 = _mm_add_ps(c4, _mm_mul_ps(a4, _mm_load1_ps((B + j+3 + m*(k+3)))));

              //Accumulate sum in C
              // #pragma omp critical
              {
              _mm_storeu_ps(c1_addr, _mm_add_ps(c1, _mm_loadu_ps(c1_addr)));
              _mm_storeu_ps(c2_addr, _mm_add_ps(c2, _mm_loadu_ps(c2_addr)));
              _mm_storeu_ps(c3_addr, _mm_add_ps(c3, _mm_loadu_ps(c3_addr)));
              _mm_storeu_ps(c4_addr, _mm_add_ps(c4, _mm_loadu_ps(c4_addr)));
              }
            }
          }
        }
      }
    }
  }
    // The following commented code is for testing purpose
    // if(0){
    //   for (int x = 0; x<3; x++){
    //     printf("|");
    //     for (int y = 0; y<3; y++){
    //       printf("%.2f|", C[y*m_a+x]);
    //     }
    //     printf("\n");
    //   }
    // }
    
    //Un-pad the matrix by transferring useful data back to original C's memory location
    if (padded){
      // #pragma omp parralel for
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


