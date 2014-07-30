#include "KRedux.hxx"

using namespace splash;

Shape
KRedux::shape(size_t N) {
  
  //The local execution range is the maximum square (2 dimensions)
  size_t 
    max_local = ocl::get().gpu.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
    L0 = sqrt(max_local),
    L1 = L0;

  //The total number of threads is the size of the input divided by the
  //number of elements each thread will consume.
  size_t total_thds = ceil(N / (float)ocl::get().ipt);
  
  //Turn the total number of threads into a square allocation
  size_t
    G0 = sqrt(total_thds),
    G1 = G0;

  //Ensure that each dimension of the square is a multiple of the
  //corresponding local execution range (OpenCL requirement for
  //defined behavior).
  G0 += (L0 - G0 % L0);
  G1 += (L1 - G1 % L1);
  
  return {{G0, G1}, {L0, L1}};
  
}
  
size_t 
KRedux::lmemSize(Shape &shp) {
  
  //The local memory is computed as the size of the workgroup. Each
  //thread within a workgroup reduces @elem_per_pe elements onto a single
  //point in the LDS, thus the LDS must be the size of the workgroup
  return shp.L[0] * shp.L[1];

}
