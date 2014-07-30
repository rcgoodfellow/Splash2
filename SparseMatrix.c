#include "SparseMatrix.h"

SparseMatrix* create_EmptySparseMatrix(unsigned long N, unsigned long n)
{
  SparseMatrix *sm = (SparseMatrix*)malloc(sizeof(SparseMatrix));
  sm->N = N;
  sm->n = n;
  sm->row_sizes = (unsigned long*)malloc(sizeof(unsigned long)*N);
  memset(sm->row_sizes, 0, N * sizeof(unsigned long));
  sm->indices = (unsigned long*)malloc(sizeof(unsigned long)*N*n);
  memset(sm->indices, -1, N * n * sizeof(unsigned long));
  sm->values = (double*)malloc(sizeof(double)*N*n);
  memset(sm->values, 0, N * n * sizeof(double));
  return sm;
}

void destroy_SparseMatrix(SparseMatrix *M)
{
  free(M->row_sizes);
  M->row_sizes = NULL;
  free(M->indices);
  M->indices = NULL;
  free(M->values);
  M->values = NULL;
}

void sm_print(SparseMatrix *M)
{
  for(unsigned long i=0; i<M->N; ++i)
  {
    unsigned long rsz = M->row_sizes[i];
    printf("%lu,%lu\n", i, rsz);
    printf("[");
    for(unsigned long j=0; j<rsz; ++j)
    {
      printf("%lu,", M->indices[i * M->n + j]);
    }
    printf("]\n");

    printf("[");
    for(unsigned long j=0; j<rsz; ++j)
    {
      printf("%f,", M->values[i * M->n + j]);
    }
    printf("]\n");
  }
}

unsigned long find(unsigned long *begin, unsigned long *end, unsigned long val)
{
  if(begin == end) { return 0; }
  unsigned long *start = begin;
  long pi = (end - begin) / 2;
  unsigned long *p = (unsigned long *)(begin + pi);

  while(end - begin > 1)
  {
    if(val <= *p) { end = p; }
    else { begin = p; }
    pi = (end - begin) / 2;
    p = (unsigned long*)(begin + pi);
  }

  if(val > *p){ ++p; }
  return p - start;
}

void mrshift(byte *begin, byte *end, unsigned long count, unsigned long size)
{
  for(byte *i = end; i >= begin; --i)
  {
    *(i + count * size) = *i;
  }
}

void mset(byte *data, unsigned long size, byte *tgt)
{
  for(unsigned long i=0; i<size; ++i) { tgt[i] = data[i]; }
}

void insert(byte *begin, byte *end, byte *val, unsigned long offset, 
    unsigned long size)
{
  mrshift(begin + offset * size, end, 1, size);
  mset(val, size, begin + offset * size);
}


void sm_set(SparseMatrix *M, unsigned long row, unsigned long col, double val)
{
  unsigned long *rb = &M->indices[row * M->n],
               *re = rb + M->row_sizes[row];
  double         *vb = &M->values[row * M->n],
               *ve = vb + M->row_sizes[row];

  unsigned long idx = find(rb, re, col);
  if(M->indices[M->n * row + idx] == col) { M->values[row * M->n + idx] = val; }
  else 
  { 
    insert((byte*)rb, (byte*)re, (byte*)&col, idx, sizeof(unsigned long));
    insert((byte*)vb, (byte*)ve, (byte*)&val, idx, sizeof(double));
    ++(M->row_sizes[row]);
  }
}


