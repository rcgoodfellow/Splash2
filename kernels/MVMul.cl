
kernel
void
tmvm(
  global double *A,
  global double *x,
  unsigned int M,
  unsigned int N,
  ushort CPG,
  ushort IPT, 
  local double *L,
  global double *Ax
   ) {

  size_t gid = get_global_id(0),
         lid = get_local_id(0),
         grid = get_group_id(0),
         lc = lid % CPG,
         lr = floor(lid / (float)CPG),
         gc = grid*CPG+lc,
         gr = lr*IPT;

  size_t as = gc + gr*N,
         xs = gr;

  double s = 0;
  for(size_t i=0; i<IPT; ++i) { s = fma(A[as+(i*N)] , x[xs+i], s); }
  L[lid] = s;

  barrier(CLK_LOCAL_MEM_FENCE);

  if(lr == 0) {
    s = 0;
    for(size_t i=0; i<256/CPG; ++i) { s += L[lc+(i*CPG)]; }
    Ax[gc] = s;
  }

}


kernel
void
tmvm_simple(
    global double *A,
    global double *x,
    unsigned int M,
    unsigned int N,
    global double *Ax
    ) {

  size_t c = (get_global_id(0) % N);

  double4 s = (double4)(0);

  size_t k = N/4;

  for(size_t i=0; i<M; ++i) { 

    double4 a = vload4(c+i*k, A);
    s = fma(a, x[i], s);
  
  }
  vstore4(s, c, Ax);
}
