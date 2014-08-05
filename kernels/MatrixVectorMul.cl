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

kernel
void
smv_mul(
    unsigned int M,
    unsigned int N,
    unsigned short n,
    global double *A,
    global unsigned int *C,
    global unsigned short *R,
    global double *x,
    global double *Ax) {

  size_t tid = get_global_id(0);
  if(tid > M){ return; }

  size_t ri = tid*n,
         rs = R[tid];
  
  Ax[tid] = 0;
  for(size_t i=0; i<rs; ++i) {

    Ax[tid] += A[ri + i] * x[C[ri+i]]; //fused multiply add candidate?

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

__kernel
void
sxvx_mul(
    __global double *S,
    __global double *x,
    ulong off,
    ulong M,
    ulong N,
    ulong NA,
    __global double *Sx) {

  size_t tid = get_global_id(0);
  if(tid >= N) { return; }

  Sx[tid] = 0.0;
  for(size_t i=0; i<M; ++i) { Sx[tid] += S[NA*i + tid] * x[i+off]; }

}
