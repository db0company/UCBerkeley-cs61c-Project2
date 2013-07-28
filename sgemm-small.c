#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>


/* ************************************************************************* */
/* Tools and Constants                                                       */
/* ************************************************************************* */
#define SPECIAL 36
#define total(width, height) (width * height)
static int block_size = 4;

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
/* Case Splitter                                                             */
/* ************************************************************************* */
void sgemmRegular(int m, int n, float * A, float * B, float * C);
void sgemmSpecial(float * A, float * B, float * C);

void sgemm(int m_a, int n_a, float * A, float * B, float * C) {
  m_a == SPECIAL && n_a == SPECIAL ?
    sgemmSpecial(A, B, C)
    : sgemmRegular(m_a, n_a, A, B, C);
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
*/                                                                                                                                               
/* ************************************************************************* */


/* ************************************************************************* */
/* Deals with general case where C = A x B, A is mxn and B is nxm            */
/* ************************************************************************* */
void sgemmRegular(int m, int n, float * A, float * B, float * C) {

  float* original_C = C;
  bool padded = false;
  int old_m;
  int old_n;

  //Pad the matrix with zeros if necessary
  if (m % block_size || n % block_size){

    padded = true;
    old_m = m;
    old_n = n;

    //Padded sizes are to the nearest multiple of block_size
    int padded_m = ((m -1) / block_size +1) * block_size;
    int padded_n = ((n -1) / block_size +1) * block_size;

    //New Matrix A, column major
    float *A_padded = (float*)calloc(total(padded_m, padded_n), sizeof(float));
    if (!A_padded) errorAllocation();
    //New Matrix B, row major
    float *B_padded = (float*)calloc(total(padded_m, padded_n), sizeof(float));
    if (!B_padded) errorAllocation();

    //Transfer data over from old matrices
    for (int n_i = 0; n_i < n; ++n_i){
      memcpy(A_padded+n_i*padded_m, A+n_i*old_m, old_m*sizeof(float));
      memcpy(B_padded+n_i*padded_m, B+n_i*old_m, old_m*sizeof(float));
    }

    //Resize matrix C, data not important
    float *C_padded = (float*)calloc(total(padded_m, padded_m), sizeof(float));
    if (!C_padded) errorAllocation();

    //reflect changes
    A = A_padded;
    B = B_padded;
    C = C_padded;
    m = padded_m;
    n = padded_n;
  }

  //Ready to multiply: C = A x B

  //Pre-declare placeholders for columns of 4x4 C_sub
  __m128 c1;
  __m128 c2;
  __m128 c3;
  __m128 c4;

  for (int k = 0; k < n; k+=4){
    //pre-calculate num of entries for row/column skipping
    int m_x_k_add_0 = m*(k+0);
    int m_x_k_add_1 = m*(k+1);
    int m_x_k_add_2 = m*(k+2);
    int m_x_k_add_3 = m*(k+3);
    
    for (int j = 0; j < m; j+=4){
      //pre-calculate the n-th entry for B and C
      float* B_add_j_add_0 = B + j+0;
      float* B_add_j_add_1 = B + j+1;
      float* B_add_j_add_2 = B + j+2;
      float* B_add_j_add_3 = B + j+3;

      //pre-load only 8 entries of 4x4 B-sub (16 xmm registers limit, other 8 used by others)
      __m128 b11 = _mm_load1_ps((B_add_j_add_0 + m_x_k_add_0));
      __m128 b12 = _mm_load1_ps((B_add_j_add_1 + m_x_k_add_0));
      __m128 b13 = _mm_load1_ps((B_add_j_add_2 + m_x_k_add_0));
      __m128 b14 = _mm_load1_ps((B_add_j_add_3 + m_x_k_add_0));

      __m128 b21 = _mm_load1_ps((B_add_j_add_0 + m_x_k_add_1));
      __m128 b22 = _mm_load1_ps((B_add_j_add_1 + m_x_k_add_1));
      __m128 b23 = _mm_load1_ps((B_add_j_add_2 + m_x_k_add_1));
      __m128 b24 = _mm_load1_ps((B_add_j_add_3 + m_x_k_add_1));

      for (int i = 0; i < m; i+=4){
        //pre-calculate the addresses for columns of C
        float* c1_addr = C + i + m*(j+0);
        float* c2_addr = C + i + m*(j+1);
        float* c3_addr = C + i + m*(j+2);
        float* c4_addr = C + i + m*(j+3);

        //Obtain placeholders for summation
        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        c4 = _mm_setzero_ps();

        //Obtain grouped single precision of 4 from columns a1, a2, a3, a4
        __m128 a1 = _mm_loadu_ps(A + i + m_x_k_add_0);
        __m128 a2 = _mm_loadu_ps(A + i + m_x_k_add_1);
        __m128 a3 = _mm_loadu_ps(A + i + m_x_k_add_2);
        __m128 a4 = _mm_loadu_ps(A + i + m_x_k_add_3);

        //To see the full formula on how positions of entries of B are calculated, refer to
        //bottom of this file

        /*b1x x a1, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, _mm_mul_ps(a1, b11));
        c2 = _mm_add_ps(c2, _mm_mul_ps(a1, b12));
        c3 = _mm_add_ps(c3, _mm_mul_ps(a1, b13));
        c4 = _mm_add_ps(c4, _mm_mul_ps(a1, b14));

        /*b2x x a2, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, _mm_mul_ps(a2, b21));
        c2 = _mm_add_ps(c2, _mm_mul_ps(a2, b22));
        c3 = _mm_add_ps(c3, _mm_mul_ps(a2, b23));
        c4 = _mm_add_ps(c4, _mm_mul_ps(a2, b24));

        /*b3x x a3, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, _mm_mul_ps(a3, _mm_load1_ps((B_add_j_add_0 + m_x_k_add_2))));
        c2 = _mm_add_ps(c2, _mm_mul_ps(a3, _mm_load1_ps((B_add_j_add_1 + m_x_k_add_2))));
        c3 = _mm_add_ps(c3, _mm_mul_ps(a3, _mm_load1_ps((B_add_j_add_2 + m_x_k_add_2))));
        c4 = _mm_add_ps(c4, _mm_mul_ps(a3, _mm_load1_ps((B_add_j_add_3 + m_x_k_add_2))));

        /*b4x x a4, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, _mm_mul_ps(a4, _mm_load1_ps((B_add_j_add_0 + m_x_k_add_3))));
        c2 = _mm_add_ps(c2, _mm_mul_ps(a4, _mm_load1_ps((B_add_j_add_1 + m_x_k_add_3))));
        c3 = _mm_add_ps(c3, _mm_mul_ps(a4, _mm_load1_ps((B_add_j_add_2 + m_x_k_add_3))));
        c4 = _mm_add_ps(c4, _mm_mul_ps(a4, _mm_load1_ps((B_add_j_add_3 + m_x_k_add_3))));

        //Accumulate sum in C
        _mm_storeu_ps(c1_addr, _mm_add_ps(c1, _mm_loadu_ps(c1_addr)));
        _mm_storeu_ps(c2_addr, _mm_add_ps(c2, _mm_loadu_ps(c2_addr)));
        _mm_storeu_ps(c3_addr, _mm_add_ps(c3, _mm_loadu_ps(c3_addr)));
        _mm_storeu_ps(c4_addr, _mm_add_ps(c4, _mm_loadu_ps(c4_addr)));

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
    for (int n_i = 0; n_i < old_m; ++n_i){
      memcpy(original_C+n_i*old_m, C+n_i*m, old_m*sizeof(float));
    }
  }
}


/* ************************************************************************* */
/* Deals with the SPECIAL case where matrix A and B are SPECIAL x SPECIAL.   */
/* ************************************************************************* */

void sgemmSpecial(float * A, float * B, float * C) {
  //Implementation currently at 12.6Gflop/s

  //Pre-declare placeholders for columns of 4x4 C_sub
  __m128 c1;
  __m128 c2;
  __m128 c3;
  __m128 c4;
  
  for (int i = 0; i < 36; i+=4){
    for (int j = 0; j < 36; j+=4){

      //Obtain placeholders for sums of columns c1, c2, c3, c4
      c1 = _mm_setzero_ps();
      c2 = _mm_setzero_ps();
      c3 = _mm_setzero_ps();
      c4 = _mm_setzero_ps();

      for (int k = 0; k < 36; k+=4){

        //Obtain grouped single precision of 4 from a1, a2, a3, a4
        __m128 a1 = _mm_loadu_ps(A + i + 36*(k+0));
        __m128 a2 = _mm_loadu_ps(A + i + 36*(k+1));
        __m128 a3 = _mm_loadu_ps(A + i + 36*(k+2));
        __m128 a4 = _mm_loadu_ps(A + i + 36*(k+3));

        /*b1x x a1, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+0 + 36*(k+0)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+1 + 36*(k+0)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+2 + 36*(k+0)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+3 + 36*(k+0)))));

        /*b2x x a2, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+0 + 36*(k+1)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+1 + 36*(k+1)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+2 + 36*(k+1)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+3 + 36*(k+1)))));

        /*b3x x a3, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+0 + 36*(k+2)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+1 + 36*(k+2)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+2 + 36*(k+2)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+3 + 36*(k+2)))));

        /*b4x x a4, for formula explanation, refer to bottom of this file*/
        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+0 + 36*(k+3)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+1 + 36*(k+3)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+2 + 36*(k+3)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+3 + 36*(k+3)))));

      }
      //Accumulate sums in C
      _mm_storeu_ps(C + i + 36*(j+0), _mm_add_ps(c1, _mm_loadu_ps(C + i + 36*(j+0))));
      _mm_storeu_ps(C + i + 36*(j+1), _mm_add_ps(c2, _mm_loadu_ps(C + i + 36*(j+1))));
      _mm_storeu_ps(C + i + 36*(j+2), _mm_add_ps(c3, _mm_loadu_ps(C + i + 36*(j+2))));
      _mm_storeu_ps(C + i + 36*(j+3), _mm_add_ps(c4, _mm_loadu_ps(C + i + 36*(j+3))));
    }
  }
}


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


