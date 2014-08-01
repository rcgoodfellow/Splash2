__kernel
void
vx_set(
    __global double *dst,
    __global double *src,
    ulong N,
    ulong ipt) {

  size_t tid = get_global_id(0);
  if(tid >= N) { return; }

  size_t begin = tid*ipt, end = min(begin + ipt, N);

  for(size_t i=begin; i<end; ++i) { dst[i] = src[i]; }

}

__kernel
void
vx_uset(
    __global double *dst,
    double value,
    ulong N,
    ulong ipt) {

  size_t tid = get_global_id(0);
  if(tid >= N) { return; }
  
  size_t begin = tid*ipt, end = min(begin + ipt, N);
  
  for(size_t i=begin; i<end; ++i) { dst[i] = value; }

}

__kernel
void
vx_sset(
    __global double *dst,
    __global double *src,
    ulong N,
    ulong dst_off,
    ulong src_off,
    ulong ipt) {

  size_t tid = get_global_id(0);
  if(tid >= N) { return; }

  size_t begin = tid*ipt, end = min(begin + ipt, N);

  for(size_t i=begin; i<end; ++i) { dst[dst_off+i] = src[src_off+i]; }

}

