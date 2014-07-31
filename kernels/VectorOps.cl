__kernel
void
vx_set(
    __global double *dst,
    __global double *src,
    ulong N,
    ulong ipt) {

  ulong tid = get_global_id(0);
  if(tid >= N) { return; }

  size_t begin = tid*ipt, end = min(begin + ipt, N);

  for(ulong i=begin; i<end; ++i) { dst[i] = src[i]; }

}
