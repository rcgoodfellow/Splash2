__kernel 
void 
kmul_mv(
    unsigned long n,
    unsigned long N,
    __global double *sm_values, 
    __global unsigned long *row_sizes,
    __global unsigned long *indices,
    __global double *dv_values,
    __global double *mv_values)
{
  int tid = get_global_id(0);
  if(tid > N) { return; }
  int ri = tid * n, 
      rs = row_sizes[tid];

  mv_values[tid] = 0;
  for(int i=0; i<rs; ++i)
  {
      mv_values[tid] += sm_values[ri + i] * dv_values[indices[ri + i]];
  }
}
