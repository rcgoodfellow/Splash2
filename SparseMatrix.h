#ifndef _SPLASH_SPARSEMATRIX_H
#define _SPLASH_SPARSEMATRIX_H

typedef char byte;

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

typedef struct SparseMatrix
{
  unsigned long N, n; //N = rows, n = max row valence
  unsigned long *row_sizes;
  unsigned long *indices;
  double *values;
}
SparseMatrix;

#ifdef __cplusplus
extern "C"
#endif
SparseMatrix* create_EmptySparseMatrix(unsigned long N, unsigned long n);
void destroy_SparseMatrix(SparseMatrix *M);

#ifdef __cplusplus
extern "C"
#endif
void sm_print(SparseMatrix *M);

#ifdef __cplusplus
extern "C"
#endif
void sm_set(SparseMatrix *M, unsigned long row, unsigned long col, double val);

unsigned long find(unsigned long *begin, unsigned long *end, unsigned long val);
void insert(byte *begin, byte *end, byte *val, unsigned long offset, 
    unsigned long size);
void mrshift(byte *begin, byte *end, unsigned long count, unsigned long size);
void mset(byte *data, unsigned long size, byte *tgt);

#endif
