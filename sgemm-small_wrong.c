
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
// __m128i load(float * m, int position) {
//   //return _mm_loadu_pd(m + position);
// }

// void store(__m128i data, float * m, int position) {
//   //_mm_storeu_pd(m + position, data);
// }

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
  // free(*A); free(*B);
  *A = A_padded;
  *B = B_padded;
  // *C = (float*)realloc(*C, total(padded_m, padded_m)*sizeof(float));
  *C = (float*) calloc(total(padded_m, padded_m), sizeof(float));

  *m_a = padded_m;
  *n_a = padded_n;
}


void unPadMatrix(int old_m, float *PaddedC, float* original_C){
  for (int n = 0; n < old_m; ++n){
    for (int m = 0; m < old_m; ++m){
      //need optimization
      original_C[m+n*old_m] = PaddedC[m+n*old_m];
    }
  }
}

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

  float* original_C = C;
  int old_m = m_a;

  // padMatrix(&m_a, &n_a, &A, &B, &C);

  for (int n = 0; n < n_a; n+=REG_BLOCKSIZE){
    for (int m = 0; m < m_a; m+=REG_BLOCKSIZE){
                                                                                                                                                                                 

      //Register Blocking, loop unrolling
      __m128 c1 = _mm_setzero_ps();
      __m128 c2 = _mm_setzero_ps();
      __m128 c3 = _mm_setzero_ps();
      __m128 c4 = _mm_setzero_ps();


      //Gets the columns a1, a2, a3, a4
      __m128 a1 = _mm_loadu_ps(A + m + m_a*(n+0));
      __m128 a2 = _mm_loadu_ps(A + m + m_a*(n+1));
      __m128 a3 = _mm_loadu_ps(A + m + m_a*(n+2));
      __m128 a4 = _mm_loadu_ps(A + m + m_a*(n+3));

      printIntel(c1);
      printIntel(c2);
      printIntel(c3);
      printIntel(c4);
      printf("\n");
      printf("\n%s\n", "Matrix A");
      printIntel(a1);
      printIntel(a2);
      printIntel(a3);
      printIntel(a4);
      printf("\n%s\n", "Matrix B");
      //Gets the rows b1, b2, b3, b4
    
      // Do not delete this reference comment block

      // __m128i b1 = _mm_loadu_ps(B + m + m_a*(n+0));
      // __m128i b2 = _mm_loadu_ps(B + m + m_a*(n+1));
      // __m128i b3 = _mm_loadu_ps(B + m + m_a*(n+2));
      // __m128i b4 = _mm_loadu_ps(B + m + m_a*(n+3));

      float* b11 = (B + m+0 + m_a*(n+0));
      float* b12 = (B + m+1 + m_a*(n+0));
      float* b13 = (B + m+2 + m_a*(n+0));
      float* b14 = (B + m+3 + m_a*(n+0));

      float* b21 = (B + m+0 + m_a*(n+1));
      float* b22 = (B + m+1 + m_a*(n+1));
      float* b23 = (B + m+2 + m_a*(n+1));
      float* b24 = (B + m+3 + m_a*(n+1));

      float* b31 = (B + m+0 + m_a*(n+2));
      float* b32 = (B + m+1 + m_a*(n+2));
      float* b33 = (B + m+2 + m_a*(n+2));
      float* b34 = (B + m+3 + m_a*(n+2));

      float* b41 = (B + m+0 + m_a*(n+3));
      float* b42 = (B + m+1 + m_a*(n+3));
      float* b43 = (B + m+2 + m_a*(n+3));
      float* b44 = (B + m+3 + m_a*(n+3));

      printIntel(_mm_loadu_ps(b11));
      printIntel(_mm_loadu_ps(b21));
      printIntel(_mm_loadu_ps(b31));
      printIntel(_mm_loadu_ps(b41));
      /*b1 x a1*/
      // c1 += 4xb11 * a1
      // c2 += 4xb12 * a1
      // c3 += 4xb13 * a1
      // c4 += 4xb14 * a1

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a1, _mm_load1_ps(b11)));
      printf("%f\n", *(b11));
      printIntel(a1);
      printIntel(c1);

      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a1, _mm_load1_ps(b12)));
      printf("%f\n", *(b12));
      printIntel(a1);
      printIntel(c2);

      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a1, _mm_load1_ps(b13)));
      printf("%f\n", *(b13));
      printIntel(a1);
      printIntel(c3);

      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a1, _mm_load1_ps(b14)));
      printf("%f\n", *(b14));
      printIntel(a1);
      printIntel(c4);

      /*b2 x a2*/
      // c1 += 4xb21 * a2
      // c2 += 4xb22 * a2
      // c3 += 4xb23 * a2
      // c4 += 4xb24 * a2
      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a2, _mm_load1_ps(b21)));
      printf("%f\n", *(b21));
      printIntel(a2);
      printIntel(c1);

      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a2, _mm_load1_ps(b22)));
      printf("%f\n", *(b22));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a2, _mm_load1_ps(b23)));
      printf("%f\n", *(b23));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a2, _mm_load1_ps(b24)));
      printf("%f\n", *(b24));

      /*b3 x a3*/
      // c1 += 4xb31 * a3
      // c2 += 4xb32 * a3
      // c3 += 4xb33 * a3
      // c4 += 4xb34 * a3

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a3, _mm_load1_ps(b31)));
      printf("%f\n", *(b31));
      printIntel(a3);
      printIntel(c1);

      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a3, _mm_load1_ps(b32)));
      printf("%f\n", *(b32));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a3, _mm_load1_ps(b33)));
      printf("%f\n", *(b33));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a3, _mm_load1_ps(b34)));
      printf("%f\n", *(b34));

      /*b4 x a4*/
      // c1 += 4xb41 * a4
      // c2 += 4xb42 * a4
      // c3 += 4xb43 * a4
      // c4 += 4xb44 * a4

      c1 = _mm_add_ps(c1, 
        _mm_mul_ps(a4, _mm_load1_ps(b41)));
      printf("%f\n", *(b41));
      printIntel(a4);
      printIntel(c1);


      c2 = _mm_add_ps(c2, 
        _mm_mul_ps(a4, _mm_load1_ps(b42)));
      printf("%f\n", *(b42));
      c3 = _mm_add_ps(c3, 
        _mm_mul_ps(a4, _mm_load1_ps(b43)));
      printf("%f\n", *(b43));
      c4 = _mm_add_ps(c4, 
        _mm_mul_ps(a4, _mm_load1_ps(b44)));
      printf("%f\n", *(b44));

      _mm_storeu_ps(C + m + m_a*(m+0), c1);
      _mm_storeu_ps(C + m + m_a*(m+1), c2);
      _mm_storeu_ps(C + m + m_a*(m+2), c3);
      _mm_storeu_ps(C + m + m_a*(m+3), c4);

      printf("\n%s\n", "Matrix C");
      printIntel(c1);
      printIntel(c2);
      printIntel(c3);
      printIntel(c4);

    } //end inner for
  } //end outer for

  // unPadMatrix(old_m, C, original_C);
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

