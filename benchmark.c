#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>

void sgemm( int m, int n, float *A, float *B, float *C );

void printMatrix(float * m, int width, int height) {
  printf("  ### Print Matrix of size %dx%d\n    ", width, height);
  for (int i = 0; i < width * height; ++i) {
    printf("%.05f ", m[i]);
    if (!((i + 1) % width))
      printf("\n    ");
  }
  printf("\n");
}

float * newMatrix(int width, int height) {
  float * m = (float *)malloc(width * height * sizeof(float));
  for (int i = 0; i < width * height; i++)
    m[i] = 2 * drand48() - 1;
  return m;
}

void run_test(int width, int height) {
  int m = width, n = height;

  float *A = newMatrix(m, n);
  float *B = newMatrix(n, m);
  float *C = newMatrix(m, m);
	    
  double seconds;
  struct timeval start, end;

  printf("## Matrice A\n");
  printMatrix(A, m, n);
  printf("## Matrice B\n");
  printMatrix(B, n, m);

  gettimeofday( &start, NULL );

  sgemm(m, n, A, B, C);

  gettimeofday( &end, NULL );

  printf("## Matrice C (after sgemm)\n");
  printMatrix(C, m, m);

  seconds = end.tv_usec - start.tv_usec;
  printf( "%d by %d matrix took %f microseconds\n", width, height, seconds);

  free( C );
  free( B );
  free( A );
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
