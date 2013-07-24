#include <nmmintrin.h>

/* Naive implementation reference */

// void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
//   for( int i = 0; i < m_a; i++ ) {
//     for( int k = 0; k < n_a; k++ ) {
//       for( int j = 0; j < m_a; j++ ) {
// 	C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
//       }
//     }
//   }
// }

/*Header*/
void padMatrix(float* mat);
void sgemm36(int m_a, int n_a, float *A, float *B, float *C);


/*Functions*/ 

/*	Deals with general case where C = A x B, A is MxN and B is NxM */
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {

}

/* deals with the special case where matrix A and B are 36x36
	requires some heavy hand-optimization to achieve 10.5Glop/s*/ 
void sgemm36(int m_a, int n_a, float *A, float *B, float *C){

}

/* Pads the matrix so that both dimentions are the nearest multiples of four*/
void padMatrix(float* mat){
}