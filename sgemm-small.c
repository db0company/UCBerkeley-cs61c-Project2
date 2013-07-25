
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

/* Transform a regular float matrix into a SIMD matrix */
__m128i load(float * m, int position) {
  //return _mm_loadu_pd(m + position);
}

void store(__m128i data, float * m, int position) {
  //_mm_storeu_pd(m + position, data);
}

// /* Add padding to the matrix so that both dimentions are the nearest
//  * multiples of four to enable register blocking.
//  * __mm_128i allows 4 single precision floating points */
// void padMatrix(int size, float ** m) {
//   if (size % REG_BLOCKSIZE) {
//     if (*m = realloc(*m, (size + (size % REG_BLOCKSIZE)) * sizeof(float)))
//       errorAllocation();
//     bzero(m + size, (size % REG_BLOCKSIZE) * sizeof(float)); /* fill with 0 */
//   }
// }

/* Add padding to the matrix so that both dimentions are the nearest
 * multiples of four to enable register blocking.
 * __mm_128i allows 4 single precision floating points */

/* Change Log
  1. change to have padMatrix do all jobs at once for performance,
    decrease function call overhead
*/
void padMatrix(int *m_a, int *n_a, float **A, float **B, float **C){

  if (!(*m_a % REG_BLOCKSIZE) && !(*n_a % REG_BLOCKSIZE)) return;

  int padded_m = ((*m_a -1) / REG_BLOCKSIZE +1) * REG_BLOCKSIZE;
  int padded_n = ((*n_a -1) / REG_BLOCKSIZE +1) * REG_BLOCKSIZE;

  //Pad Matrix A, column major
  float *A_padded = (float*)calloc(total(padded_m, padded_n), sizeof(float));
  if (!A_padded) errorAllocation();

  for (int n = 0; n < *n_a; ++n){
    for (int m = 0; m < *m_a; ++m){
      //need optimization
      A_padded[m+n*padded_m] = (*A)[m+n*(*m_a)];
    }
  }

  //Pad Matrix B, row major
  float *B_padded = (float*)calloc(total(padded_m, padded_n), sizeof(float));
  for (int m = 0; m < *m_a; ++m){
    for (int n = 0; n < *n_a; ++n){
      //need optimization
      B_padded[n+m*padded_n] = (*B)[n+m*(*n_a)];
    }
  }

  //reflect changes
  free(*A); free(*B);
  *A = A_padded;
  *B = B_padded;
  *C = (float*)realloc(C, total(padded_m, padded_m)*sizeof(float));
  *m_a = padded_m;
  *n_a = padded_n;
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
  __m128i r;

  padMatrix(&m_a, &n_a, &A, &B, &C);

  for (int leap_n = 0; leap_n < n_a; leap_n+=REG_BLOCKSIZE){
    for (int leap_m = 0; leap_m < m_a; leap_m+=REG_BLOCKSIZE){
                                                                                                                                                                                 

      //Register Blocking, loop unrolling
      __m128 c1 = _mm_setzero_ps();
      __m128 c2 = _mm_setzero_ps();
      __m128 c3 = _mm_setzero_ps();
      __m128 c4 = _mm_setzero_ps();

      //Gets the columns a1, a2, a3, a4
      __m128 a1 = _mm_loadu_ps(A + leap_m + m_a*(leap_n+0));
      __m128 a2 = _mm_loadu_ps(A + leap_m + m_a*(leap_n+1));
      __m128 a3 = _mm_loadu_ps(A + leap_m + m_a*(leap_n+2));
      __m128 a4 = _mm_loadu_ps(A + leap_m + m_a*(leap_n+3));

      //Gets the rows b1, b2, b3, b4
/*    
      Do not delete this reference comment block

      // __m128i b1 = _mm_loadu_ps(B + leap_m + m_a*(leap_n+0));
      // __m128i b2 = _mm_loadu_ps(B + leap_m + m_a*(leap_n+1));
      // __m128i b3 = _mm_loadu_ps(B + leap_m + m_a*(leap_n+2));
      // __m128i b4 = _mm_loadu_ps(B + leap_m + m_a*(leap_n+3));

      float b11 = *(B + leap_m+0 + m_a*(leap_n+0));
      float b12 = *(B + leap_m+1 + m_a*(leap_n+0));
      float b13 = *(B + leap_m+2 + m_a*(leap_n+0));
      float b14 = *(B + leap_m+3 + m_a*(leap_n+0));

      float b21 = *(B + leap_m+0 + m_a*(leap_n+1));
      float b22 = *(B + leap_m+1 + m_a*(leap_n+1));
      float b23 = *(B + leap_m+2 + m_a*(leap_n+1));
      float b24 = *(B + leap_m+3 + m_a*(leap_n+1));

      float b31 = *(B + leap_m+0 + m_a*(leap_n+2));
      float b32 = *(B + leap_m+1 + m_a*(leap_n+2));
      float b33 = *(B + leap_m+2 + m_a*(leap_n+2));
      float b34 = *(B + leap_m+3 + m_a*(leap_n+2));

      float b41 = *(B + leap_m+0 + m_a*(leap_n+3));
      float b42 = *(B + leap_m+1 + m_a*(leap_n+3));
      float b43 = *(B + leap_m+2 + m_a*(leap_n+3));
      float b44 = *(B + leap_m+3 + m_a*(leap_n+3));
*/
      /*b1 x a1*/
      // c1 += 4xb11 * a1
      // c2 += 4xb12 * a1
      // c3 += 4xb13 * a1
      // c4 += 4xb14 * a1

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a1, _mm_load1_ps(B + leap_m+0 + m_a*(leap_n+0))));
      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a1, _mm_load1_ps(B + leap_m+1 + m_a*(leap_n+0))));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a1, _mm_load1_ps(B + leap_m+2 + m_a*(leap_n+0))));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a1, _mm_load1_ps(B + leap_m+3 + m_a*(leap_n+0))));

      /*b2 x a2*/
      // c1 += 4xb21 * a2
      // c2 += 4xb22 * a2
      // c3 += 4xb23 * a2
      // c4 += 4xb24 * a2
      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a2, _mm_load1_ps(B + leap_m+0 + m_a*(leap_n+1))));
      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a2, _mm_load1_ps(B + leap_m+1 + m_a*(leap_n+1))));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a2, _mm_load1_ps(B + leap_m+2 + m_a*(leap_n+1))));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a2, _mm_load1_ps(B + leap_m+3 + m_a*(leap_n+1))));

      /*b3 x a3*/
      // c1 += 4xb31 * a3
      // c2 += 4xb32 * a3
      // c3 += 4xb33 * a3
      // c4 += 4xb34 * a3

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a3, _mm_load1_ps(B + leap_m+0 + m_a*(leap_n+2))));
      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a3, _mm_load1_ps(B + leap_m+1 + m_a*(leap_n+2))));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a3, _mm_load1_ps(B + leap_m+2 + m_a*(leap_n+2))));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a3, _mm_load1_ps(B + leap_m+3 + m_a*(leap_n+2))));

      /*b4 x a4*/
      // c1 += 4xb41 * a4
      // c2 += 4xb42 * a4
      // c3 += 4xb43 * a4
      // c4 += 4xb44 * a4

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a4, _mm_load1_ps(B + leap_m+0 + m_a*(leap_n+3))));
      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a4, _mm_load1_ps(B + leap_m+1 + m_a*(leap_n+3))));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a4, _mm_load1_ps(B + leap_m+2 + m_a*(leap_n+3))));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a4, _mm_load1_ps(B + leap_m+3 + m_a*(leap_n+3))));

      _mm_storeu_ps(C + leap_m + m_a*(leap_n+0), c1);
      _mm_storeu_ps(C + leap_m + m_a*(leap_n+1), c1);
      _mm_storeu_ps(C + leap_m + m_a*(leap_n+2), c1);
      _mm_storeu_ps(C + leap_m + m_a*(leap_n+3), c1);

    } //end inner for
  } //end outer for
}

/* Deals with the SPECIAL case where matrix A and B are SPECIAL x SPECIAL.
 * Requires some heavy hand-optimization to achieve 10.5Glop/s */ 
void sgemmSpecial(float * A, float * B, float * C) {
  sgemmRegular(36, 36, A, B, C);
}

void sgemm(int m_a, int n_a, float * A, float * B, float * C) {
  m_a == SPECIAL && n_a == SPECIAL ?
    sgemmSpecial(A, B, C)
    : sgemmRegular(m_a, n_a, A, B, C);
}

