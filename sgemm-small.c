
#include <nmmintrin.h>
#include <strings.h>
#include <stdio.h>

#define SPECIAL 36
#define BLOCKSIZE sizeof(__m128i) / sizeof(float) // should be 4

#define A_width  m_a
#define A_height n_a

#define B_width  n_a
#define B_height m_a

#define C_width  m_a
#define C_height m_a

/* ************************************************************************* */
/* Tools                                                                     */
/* ************************************************************************* */

#define total(width, height) (width * height)

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

/* Add padding to the matrix so that both dimentions are the nearest
 * multiples of four to enable register blocking.
 * __mm_128i allows 4 single precision floating points */
void padMatrix(int size, float ** m) {
  if (size % BLOCKSIZE) {
    if (*m = realloc(*m, (size + (size % BLOCKSIZE)) * sizeof(float)))
      errorAllocation();
    bzero(m + size, (size % BLOCKSIZE) * sizeof(float)); /* fill with 0 */
  }
}

/* ************************************************************************* */
/* SGEMM Cases                                                               */
/* ************************************************************************* */

/* Deals with general case where C = A x B, A is MxN and B is NxM */
void sgemmRegular(int m_a, int n_a, float * A, float * B, float * C) {
  __m128i r;
  padMatrix(total(A_width, A_height), &A);
  padMatrix(total(B_width, B_height), &B);
  padMatrix(total(C_width, C_height), &C);
  for (int i = 0; i < total(C_width, C_height); i += BLOCKSIZE) {
    // todo: algorithm
    //r = _mm_add_pd(load(C, i), _mm_mul_pd(load(A, i), load(B, i)));
    //store(r, C, i);
  }
}

/* Deals with the SPECIAL case where matrix A and B are SPECIAL x SPECIAL.
 * Requires some heavy hand-optimization to achieve 10.5Glop/s */ 
void sgemmSpecial(float * A, float * B, float * C) {
  // todo
}

void sgemm(int m_a, int n_a, float * A, float * B, float * C) {
  m_a == SPECIAL && n_a == SPECIAL ?
    sgemmSpecial(A, B, C)
    : sgemmRegular(m_a, n_a, A, B, C);
}

