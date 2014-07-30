__kernel
void
mx_colset(
    __global double *A,
    __global double *v,
    ulong N,
    ulong M,
    ulong cidx,
    ulong ipt) {

  ulong tid = get_global_id(0);
  if(tid >= N) { return; }

  for(ulong i=0; i<ipt; ++i) { 
    if(tid*ipt+i > N*M){ break; }
    A[M*(tid*ipt+i)] = v[tid*ipt+i]; 
  }

}

__kernel
void
mx_zero(
    __global double *A,
    ulong N,
    ulong M,
    ulong ipt) {

  ulong tid = get_global_id(0);
  if(tid >= N) { return; }

  size_t begin = tid*ipt, end = min(begin + ipt, N*M);

  for(ulong i=begin; i<end; ++i) { A[i] = 0; }

}

