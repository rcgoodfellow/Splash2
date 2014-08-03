#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel
void
sx_ident(
    __global double *S,
    ulong N,
    ulong NA,
    ulong M,
    ulong ipt) {

  size_t tid = get_global_id(0);

  size_t begin = tid*ipt, end = min(begin+ipt, M);

  for(ulong i=begin; i<end; ++i) {
    S[i*NA + i%NA] = 1;
  }
}

__kernel
void
sx_eq(
    __global double *A,
    __global double *B,
    ulong N,
    ulong ipt,
    double thresh,
    __global ulong *diff) {

  size_t tid = get_global_id(0);
  
  size_t begin = tid*ipt, end = min(begin+ipt, N);
  for(size_t i=begin; i<end; ++i) {
    if(fabs(A[i] - B[i]) > thresh) { atom_inc(diff); }
  }

}
