#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>

/* Your function must have the following signature: */

void sgemm( int m, int n, float *A, float *B, float *C );

void run_test(int width, int height) {
  for( int n = width; n <= width; n = n+1+n/3 )
    {
      /* Try different m */
      for( int m = height; m <= height; m = m+1+m/3 )
	{
	  /* Allocate and fill 2 random matrices A, C */
	  float *A = (float*) malloc( m * n * sizeof(float) );
	  float *B = (float*) malloc( n * m * sizeof(float) );
	  float *C = (float*) malloc( m * m * sizeof(float) );
	    
	  for( int i = 0; i < m*n; i++ ) A[i] = 2 * drand48() - 1;
	  for( int i = 0; i < n*m; i++ ) B[i] = 2 * drand48() - 1;
	  for( int i = 0; i < m*m; i++ ) C[i] = 2 * drand48() - 1;
	    
	  /* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
	  double Gflop_s, seconds = -1.0;
	  for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 ) 
	    {
	      /* warm-up */
	      sgemm( m, n, A, B, C );
		
	      /* measure time */
	      struct timeval start, end;
	      gettimeofday( &start, NULL );
	      for( int i = 0; i < n_iterations; i++ )
		sgemm( m,n, A, B, C );
	      gettimeofday( &end, NULL );
	      seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
      
	      /* compute Gflop/s rate */
	      Gflop_s = 2e-9 * n_iterations * m * m * n / seconds;
	    }
    
	  printf( "%d by %d matrix \t %g Gflop/s\n", m, n, Gflop_s );
    
	  /* Ensure that error does not exceed the theoretical error bound */
		
	  /* Set initial C to 0 and do matrix multiply of A*B */
	  memset( C, 0, sizeof( float ) * m * m );
	  sgemm( m,n, A, B, C );

	  /* Subtract A*B from C using standard sgemm (note that this should be 0 to within machine roundoff) */
	  cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,n, -1, A,m, B,m, 1, C,m );

	  /* Subtract the maximum allowed roundoff from each element of C */
	  for( int i = 0; i < m*n; i++ ) A[i] = fabs( A[i] );
	  for( int i = 0; i < m*n; i++ ) B[i] = fabs( B[i] );
	  for( int i = 0; i < m*m; i++ ) C[i] = fabs( C[i] );
	  cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,n, -3.0*FLT_EPSILON*n, A,m, B,m, 1, C,m );

	  /* After this test if any element in C is still positive something went wrong in square_sgemm */
	  for( int i = 0; i < m * m; i++ )
	    if( C[i] > 0 ) {
	      printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
	      exit(EXIT_FAILURE);
	    }

	  /* release memory */
	  free( C );
	  free( B );
	  free( A );
	}
    }  
}

/* Search a character in a string.
 * If found return position else return -1. */
int getCharPos(char * s, char c) {
  for (int i = 0; s[i]; ++i)
    if (s[i] == c)
      return i;
  return -1;
}

/* The benchmarking program
 * Takes as a parameter as many tests you want
 * Give it a number if you want a square matrix
 * Give it two numbers separated by an 'x' for another matrix
 * Example: ./bench-small 10 3x6 5 89x2
*/

int main( int argc, char **argv )
{
  char ** to_test;
  char * default_test[] = {"36", NULL};
  int i, pos;
  char * str, * width, * height;

  to_test = (argc < 2) ? default_test : &argv[1];
  srand(time(NULL));
  for (i = 0; to_test[i]; ++i) {
    str = strdup(to_test[i]);
    // rectangle matrix
    if (pos = getCharPos(str, 'x')) {
      height = &str[pos + 1];
      str[pos] = '\0';
      width = str;
    }
    // square matrix
    else {
      width = height = str;
    }
    run_test(atoi(width), atoi(height));
    free(str);
  }
  return EXIT_SUCCESS;
}
