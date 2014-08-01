__kernel 
void 
smvx_mul(
    ulong n,
    ulong N,
    __global double *sm_values, 
    __global ulong *row_sizes,
    __global ulong *indices,
    __global double *dv_values,
    __global double *mv_values) {

  size_t tid = get_global_id(0);
  if(tid >= N) { return; }
  size_t ri = tid * n, 
         rs = row_sizes[tid];

  mv_values[tid] = 0;
  for(size_t i=0; i<rs; ++i)
  {
      mv_values[tid] += sm_values[ri + i] * dv_values[indices[ri + i]];
  }
}

__kernel
void
mxvx_mul(
    __global double *A,
    __global double *v,
    ulong off,
    ulong N,
    ulong NA,
    ulong M,
    __global double *Av) {

  size_t tid = get_global_id(0);
  if(tid >= M) { return; }

  Av[tid] = 0.0;
  for(size_t i=0; i<N; ++i) { Av[tid] += A[NA*tid + i] * v[i+off]; }
}
