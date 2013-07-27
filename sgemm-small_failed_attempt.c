
#include <nmmintrin.h>
#include <strings.h>
#include <stdio.h>

#define SPECIAL 36

#define A_height  m_a
#define A_width n_a

#define B_height  n_a
#define B_width m_a

#define C_height m_a
#define C_width  m_a

/* ************************************************************************* */
/* Tools                                                                     */
/* ************************************************************************* */

#define total(width, height) (width * height)
static int REG_BLOCKSIZE = sizeof(__m128) / sizeof(float);

void error(char * msg) {
  fprintf(stderr, "[Error] %s\n", msg);
  exit(EXIT_FAILURE);
}

/* Call me when *malloc fails */
void errorAllocation(void) {
  error("Allocation failure");
}

/* ************************************************************************* */
/* Matrix Modifications                                                      */
/* ************************************************************************* */

void printIntel(__m128 var){
  printf("{%f, %f, %f, %f}\n", 
    *((float*)(&var)+0),
    *((float*)(&var)+1),
    *((float*)(&var)+2),
    *((float*)(&var)+3));
}

/* ************************************************************************* */
/* SGEMM Cases                                                        

              
          R e g i s t e r   B l o c k i n g 
          
          a1, a2, a3, and a4 refer to matrix A
          b1, b2, b3, and b4 refer to matrix B    
          c1, c2, c3, and c4 refer to matrix C

          a 1       a 2       a 3       a 4   
          c 1       c 2       c 3       c 4                                                                                                                   
          » » » »   » » » »   » » » »   » » » »                                                                                                                  
      b 1                                                                                                                                                        
          » » » »   » » » »   » » » »   » » » »                                                                                                                  
      b 2                                                                                                                                                        
          » » » »   » » » »   » » » »   » » » »                                                                                                                  
      b 3                                                                                                                                                        
          » » » »   » » » »   » » » »   » » » »                                                                                                                  
      b 4                                                                                                                                                        
          » » » »   » » » »   » » » »   » » » »                                                                                                                  
                                                                                                                                                                 
         
*/                                                                                                                                               
/* ************************************************************************* */

/* Deals with general case where C = A x B, A is MxN and B is NxM */
void sgemmRegular(int m_a, int n_a, float * A, float * B, float * C) {

  int k, j, i;
  for (k = 0; k < n_a; k+=4){
   for (j = 0; j < m_a; j+=4){
      for (i = 0; i < m_a; i+=4){
        //Obtain placeholders for sums of column c1, c2, c3, c4
        __m128 c1 = _mm_setzero_ps();
        __m128 c2 = _mm_setzero_ps();
        __m128 c3 = _mm_setzero_ps();
        __m128 c4 = _mm_setzero_ps();

        //Obtain grouped single precision of 4 from a1, a2, a3, a4
        __m128 a1 = _mm_loadu_ps(A + i + m_a*(k+0));
        __m128 a2 = _mm_loadu_ps(A + i + m_a*(k+1));
        __m128 a3 = _mm_loadu_ps(A + i + m_a*(k+2));
        __m128 a4 = _mm_loadu_ps(A + i + m_a*(k+3));

        //Do not delete this reference
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

        /*b1 x a1*/
        // c1 += 4xb11 * a1
        // c2 += 4xb12 * a1
        // c3 += 4xb13 * a1
        // c4 += 4xb14 * a1

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+0 + m_a*(k+0)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+1 + m_a*(k+0)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+2 + m_a*(k+0)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+3 + m_a*(k+0)))));

        /*b2 x a2*/
        // c1 += 4xb21 * a2
        // c2 += 4xb22 * a2
        // c3 += 4xb23 * a2
        // c4 += 4xb24 * a2

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+0 + m_a*(k+1)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+1 + m_a*(k+1)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+2 + m_a*(k+1)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+3 + m_a*(k+1)))));

        /*b3 x a3*/
        // c1 += 4xb31 * a3
        // c2 += 4xb32 * a3
        // c3 += 4xb33 * a3
        // c4 += 4xb34 * a3

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+0 + m_a*(k+2)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+1 + m_a*(k+2)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+2 + m_a*(k+2)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+3 + m_a*(k+2)))));

        /*b4 x a4*/
        // c1 += 4xb41 * a4
        // c2 += 4xb42 * a4
        // c3 += 4xb43 * a4
        // c4 += 4xb44 * a4

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+0 + m_a*(k+3)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+1 + m_a*(k+3)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+2 + m_a*(k+3)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+3 + m_a*(k+3)))));

        //Put c's back on shelf
        _mm_storeu_ps(C + i + m_a*(j+0), _mm_add_ps(c1, _mm_loadu_ps(C + i + m_a*(j+0))));
        _mm_storeu_ps(C + i + m_a*(j+1), _mm_add_ps(c2, _mm_loadu_ps(C + i + m_a*(j+1))));
        _mm_storeu_ps(C + i + m_a*(j+2), _mm_add_ps(c3, _mm_loadu_ps(C + i + m_a*(j+2))));
        _mm_storeu_ps(C + i + m_a*(j+3), _mm_add_ps(c4, _mm_loadu_ps(C + i + m_a*(j+3))));

      }
    }
  }

  //post-optimization
  for (k = 0; k < n_a; k++){
    for (j = 0; j < m_a; j++){
      for (i = m_a/4*4; i < m_a; i++){
        C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
      }
    }
  }


  
}

/* Deals with the SPECIAL case where matrix A and B are SPECIAL x SPECIAL.
 * Requires some heavy hand-optimization to achieve 10.5Glop/s */ 
void sgemmSpecial(float * A, float * B, float * C) {
  //Implementation currently at 11Gflop/s
  int m_a = 36;
  int n_a = 36;
  for (int i = 0; i < m_a; i+=4){
    for (int j = 0; j < m_a; j+=4){
      for (int k = 0; k < n_a; k+=4){
        //Obtain placeholders for sums of column c1, c2, c3, c4
        __m128 c1 = _mm_setzero_ps();
        __m128 c2 = _mm_setzero_ps();
        __m128 c3 = _mm_setzero_ps();
        __m128 c4 = _mm_setzero_ps();

        //Obtain grouped single precision of 4 from a1, a2, a3, a4
        __m128 a1 = _mm_loadu_ps(A + i + m_a*(k+0));
        __m128 a2 = _mm_loadu_ps(A + i + m_a*(k+1));
        __m128 a3 = _mm_loadu_ps(A + i + m_a*(k+2));
        __m128 a4 = _mm_loadu_ps(A + i + m_a*(k+3));


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


        /*b1 x a1*/
        // c1 += 4xb11 * a1
        // c2 += 4xb12 * a1
        // c3 += 4xb13 * a1
        // c4 += 4xb14 * a1

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+0 + m_a*(k+0)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+1 + m_a*(k+0)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+2 + m_a*(k+0)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a1, _mm_load1_ps((B + j+3 + m_a*(k+0)))));

        /*b2 x a2*/
        // c1 += 4xb21 * a2
        // c2 += 4xb22 * a2
        // c3 += 4xb23 * a2
        // c4 += 4xb24 * a2

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+0 + m_a*(k+1)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+1 + m_a*(k+1)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+2 + m_a*(k+1)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a2, _mm_load1_ps((B + j+3 + m_a*(k+1)))));

        /*b3 x a3*/
        // c1 += 4xb31 * a3
        // c2 += 4xb32 * a3
        // c3 += 4xb33 * a3
        // c4 += 4xb34 * a3

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+0 + m_a*(k+2)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+1 + m_a*(k+2)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+2 + m_a*(k+2)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a3, _mm_load1_ps((B + j+3 + m_a*(k+2)))));

        /*b4 x a4*/
        // c1 += 4xb41 * a4
        // c2 += 4xb42 * a4
        // c3 += 4xb43 * a4
        // c4 += 4xb44 * a4

        c1 = _mm_add_ps(c1, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+0 + m_a*(k+3)))));
        c2 = _mm_add_ps(c2, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+1 + m_a*(k+3)))));
        c3 = _mm_add_ps(c3, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+2 + m_a*(k+3)))));
        c4 = _mm_add_ps(c4, 
          _mm_mul_ps(a4, _mm_load1_ps((B + j+3 + m_a*(k+3)))));

        //Put c's back on shelf
        _mm_storeu_ps(C + i + m_a*(j+0), _mm_add_ps(c1, _mm_loadu_ps(C + i + m_a*(j+0))));
        _mm_storeu_ps(C + i + m_a*(j+1), _mm_add_ps(c2, _mm_loadu_ps(C + i + m_a*(j+1))));
        _mm_storeu_ps(C + i + m_a*(j+2), _mm_add_ps(c3, _mm_loadu_ps(C + i + m_a*(j+2))));
        _mm_storeu_ps(C + i + m_a*(j+3), _mm_add_ps(c4, _mm_loadu_ps(C + i + m_a*(j+3))));

      }
    }
  }
}

void sgemm(int m_a, int n_a, float * A, float * B, float * C) {
  m_a == SPECIAL && n_a == SPECIAL ?
    sgemmSpecial(A, B, C)
    : sgemmRegular(m_a, n_a, A, B, C);
}

