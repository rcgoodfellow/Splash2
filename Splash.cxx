#include "Splash.hxx"
#include "KRedux.hxx"
#include "SparseMatrix.hxx"

using namespace splash;
using std::string;
using std::make_pair;
using std::runtime_error;
using std::to_string;
using std::vector;
using std::stringstream;

//dvec ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dvec
splash::new_dvec(std::vector<double> values)
{
  dvec v{};
  v.N = values.size();
  v.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(double) * v.N,
      values.data());

  return v;
}

dvec &
dvec::operator = (const dvec &a) {

  if(a.N != N) {
    throw runtime_error("assignment of incompatible dvecs");
  }

  cl::Kernel kvx_set{ocl::get().libsplash, "vx_set"};

  kvx_set.setArg(0, v);
  kvx_set.setArg(1, a.v);
  kvx_set.setArg(2, N);
  kvx_set.setArg(3, ocl::get().ipt);

  ocl::get().q.enqueueNDRangeKernel(
      kvx_set,
      cl::NullRange,
      cl::NDRange{ocl::gsize(N)},
      cl::NDRange{ocl::lsize()});

  return *this;
}

double*
dvec::readback() {
  
  double *r = (double*)malloc(sizeof(double)*N);
  ocl::get().q.enqueueReadBuffer(
      v,
      CL_TRUE,
      0,
      sizeof(double)*N,
      r);

  return r;
}

string splash::show_vec(double *v, size_t N) {

  stringstream ss;
  ss << "[";
  for(size_t i=0; i<N-1; ++i) {
    ss << v[i] << ","; 
  }
  ss << v[N-1] << "]";
  return ss.str();

}

string splash::show_matrix(double *A, size_t N, size_t M) { 

  stringstream ss;
  ss << std::setprecision(3) << std::fixed;
  for(size_t i=0; i<N; ++i) { 
    for(size_t j=0; j<M; ++j) {
      ss << A[M*i+j] << "\t";
    }
    ss << std::endl;
  }

  return ss.str();

}

string splash::show_subspace(double *S, size_t N, size_t M) {

  stringstream ss;
  ss << std::setprecision(3) << std::fixed;
  for(size_t i=0; i<N; ++i) {
    for(size_t j=0; j<M; ++j) {
      ss << S[N*j+i] << "\t";
    }
    ss << std::endl;
  }

  return ss.str();

}

//dsmatrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dsmatrix
splash::new_dsmatrix(vector<sm_row> elems)
{
  size_t N = elems.size();
  size_t n = 
    std::max_element(elems.begin(), elems.end(), 
      [](const sm_row &a, const sm_row &b){ 
          return a.size() < b.size(); })->size();

  SparseMatrix *sM = create_EmptySparseMatrix(N,n);
  for(size_t i=0; i<elems.size(); ++i) {
    for(size_t j=0; j<elems[i].size(); ++j) {
      sm_set(sM, i, elems[i][j].i, elems[i][j].v); }}

  dsmatrix m{};
  m.N = N;
  m.n = n;

  m.ri = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(unsigned long) * m.N,
      sM->row_sizes);

  m.ci = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(unsigned long) * m.N * m.n,
      sM->indices);

  m.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(double) * m.N * m.n,
      sM->values);

  return m;
}

double
dscalar::readback() {

  double value{0};
  ocl::get().q.enqueueReadBuffer(
      v,
      CL_TRUE,
      0,
      sizeof(double),
      &value);

  return value;
}

//dmatrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dmatrix::dmatrix(size_t N, size_t M) 
  : N{N}, M{M}, v{ocl::get().ctx, CL_MEM_READ_WRITE, sizeof(double)*N*M} { }


dmatrix::dcol dmatrix::C(size_t idx) {

  return dcol{this, idx};
}

dmatrix::dcol::dcol(dmatrix *parent, size_t idx) : parent{parent}, idx{idx} {}

dmatrix::dcol& dmatrix::dcol::operator = (const dvec &v) {

  //sanity check
  if(parent->N != v.N) { 
    throw runtime_error("Incompatible matrix column assignment"); 
  }

  cl::Kernel kmx_colset = cl::Kernel(ocl::get().libsplash, "mx_colset");

  kmx_colset.setArg(0, parent->v);
  kmx_colset.setArg(1, v.v);
  kmx_colset.setArg(2, parent->N);
  kmx_colset.setArg(3, parent->M);
  kmx_colset.setArg(4, idx);
  kmx_colset.setArg(5, ocl::get().ipt);
  
  size_t gsize = static_cast<size_t>(ceil(v.N/ocl::get().ipt));
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      kmx_colset,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

  return *this; 

}

double*
dmatrix::readback() {

  double *d = (double*)malloc(sizeof(double)*N*M);
  ocl::get().q.enqueueReadBuffer(
      v,
      CL_TRUE,
      0,
      sizeof(double)*N*M,
      d);

  return d;

}

void
dmatrix::zero() {

  cl::Kernel kmx_zero = cl::Kernel(ocl::get().libsplash, "mx_zero");

  kmx_zero.setArg(0, v);
  kmx_zero.setArg(1, N);
  kmx_zero.setArg(2, M);
  kmx_zero.setArg(3, ocl::get().ipt);

  size_t gsize = static_cast<size_t>(ceil(M*N/ocl::get().ipt));
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      kmx_zero,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

}

void
dsubspace::zero() {
  
  cl::Kernel kmx_zero = cl::Kernel(ocl::get().libsplash, "mx_zero");

  kmx_zero.setArg(0, v);
  kmx_zero.setArg(1, NA);
  kmx_zero.setArg(2, M);
  kmx_zero.setArg(3, ocl::get().ipt);

  ocl::get().q.enqueueNDRangeKernel(
      kmx_zero,
      cl::NullRange,
      cl::NDRange{ocl::gsize(NA*M)},
      cl::NDRange{ocl::lsize()});

}

#include <iostream>
dsubspace::dsubspace(size_t N, size_t M) 
  : N{N}, M{M} { 

  size_t base_align = 
    ocl::get()
      .gpu
      .getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8; //bits -> bytes

  std::cout << "base align(bytes)=" << base_align << std::endl;

  NA = N*sizeof(double);
  NA += base_align - (NA % base_align);
  NA /= sizeof(double);

  v = cl::Buffer{ocl::get().ctx, CL_MEM_READ_WRITE, sizeof(double)*NA*M};
}

dvec dsubspace::operator () (size_t idx) {

  dvec x;
  x.N = N;
  x.NA = NA;
  x.br = (_cl_buffer_region*)malloc(sizeof(_cl_buffer_region));
  x.br->origin = NA*idx*sizeof(double);
  x.br->size = NA*sizeof(double);
  
  std::cout << "origin=" << x.br->origin << std::endl;

  x.v = v.createSubBuffer(CL_MEM_READ_WRITE,
      CL_BUFFER_CREATE_TYPE_REGION,
      x.br);
  
  return x;

}

double* dsubspace::readback() {

  double *d = (double*)malloc(sizeof(double)*NA*M);
  ocl::get().q.enqueueReadBuffer(
      v,
      CL_TRUE,
      0,
      sizeof(double)*NA*M,
      d);

  return d;

}

//LibSplash ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LibSplash::LibSplash(string splashdir)
  : splashdir{splashdir} {

    build_opts = "-I " + splashdir + " -DREAL=double";
    readSource();
}

void
LibSplash::readSource() {

  redux_st = read_file(splashdir + "/kernels/Redux.cl");
  elemental_st = read_file(splashdir + "/kernels/Elementals.cl");
  mvmul_st = read_file(splashdir + "/kernels/MatrixVectorMul.cl");
  mxops_st = read_file(splashdir + "/kernels/MatrixOps.cl");
  vecops_st = read_file(splashdir + "/kernels/VectorOps.cl");

  src = {
    make_pair(redux_st.c_str(), redux_st.length()),
    make_pair(elemental_st.c_str(), elemental_st.length()),
    make_pair(mvmul_st.c_str(), mvmul_st.length()),
    make_pair(mxops_st.c_str(), mxops_st.length()),
    make_pair(vecops_st.c_str(), vecops_st.length())
  };

}

cl::Program
LibSplash::get(cl::Context ctx) {

  cl::Program libsplash(ctx, src);
  try{ 
    
    libsplash.build(build_opts.c_str()); 
  
  }
  catch(cl::Error&) {
    throw runtime_error(
        libsplash.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
          ctx.getInfo<CL_CONTEXT_DEVICES>()[0]));
  }
  
  return libsplash;
}

//utility functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

std::string splash::read_file(std::string filename) {
  std::ifstream t(filename);
  if(!t.good()) {
    t.close();
    throw runtime_error("Unable to read file: " + filename);
  }
  std::stringstream buffer;
  buffer << t.rdbuf();

  return std::string(buffer.str());
}

//operators -------------------------------------------------------------------

dscalar splash::redux_add(const dvec &v) {
  
  //create a dscalar to hold the result of the redux and register its buffer 
  //with opencl
  dscalar s{};
  s.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double));

  cl::Kernel krdx_add = cl::Kernel(ocl::get().libsplash, "redux_add");

  Shape krdx_shp = KRedux::shape(v.N);
  size_t krdx_lmem_sz = sizeof(double) * KRedux::lmemSize(krdx_shp);
  size_t krdx_result_sz = krdx_shp.wgCount();
  
  cl::Buffer krdx_result{ocl::get().ctx,
    CL_MEM_READ_WRITE,
    sizeof(double) * krdx_result_sz};

  krdx_add.setArg(0, v.v);
  krdx_add.setArg(1, v.N);
  krdx_add.setArg(2, ocl::get().ipt);
  krdx_add.setArg(3, cl::Local(krdx_lmem_sz));
  krdx_add.setArg(4, krdx_result);
  ocl::get().q.enqueueNDRangeKernel(
      krdx_add,
      cl::NullRange,
      krdx_shp.G,
      krdx_shp.L);

  //it could be (and is likely) that the redux left us with a bit to reduce
  //if so go ahead and reduce it
  if(krdx_result_sz > 1) {

    krdx_shp = KRedux::shape(v.N);
    krdx_lmem_sz = sizeof(double) * KRedux::lmemSize(krdx_shp);
    krdx_result_sz = krdx_shp.wgCount();

    cl::Buffer krdx_result_2{ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double) * krdx_result_sz};

    krdx_add.setArg(0, krdx_result);
    krdx_add.setArg(1, krdx_result_sz);
    krdx_add.setArg(2, ocl::get().ipt);
    krdx_add.setArg(3, cl::Local(krdx_lmem_sz));
    krdx_add.setArg(4, krdx_result_2);
    ocl::get().q.enqueueNDRangeKernel(
      krdx_add,
      cl::NullRange,
      krdx_shp.G,
      krdx_shp.L);

    s.v = krdx_result_2;

  }
  else{ s.v = krdx_result; }

  return s;
}

dscalar splash::operator* (const dvec &a, const dvec &b) {
  
  //sanity checks
  if(a.N != b.N) { 
    throw runtime_error("Attempt to multiply vectors of different lengths"); 
  }

  //create a dvec to hold the result of the vector-vector multiplication
  //and register its buffer with opencl
  dvec s{};
  s.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double)*a.N);
  s.N = a.N;

  //vector-vector multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //grab the vector-vector multiplication kernel from the splash library
  cl::Kernel kmul_vv = cl::Kernel(ocl::get().libsplash, "kmul_vv");

  //set the kernel arguments
  kmul_vv.setArg(0, a.v);
  kmul_vv.setArg(1, b.v);
  kmul_vv.setArg(2, a.N);
  kmul_vv.setArg(3, ocl::get().ipt);
  kmul_vv.setArg(4, s.v);

  //global work size needs to be a multiple of the local work size (256)
  size_t gsize = static_cast<size_t>(ceil(a.N/ocl::get().ipt));
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      kmul_vv,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});
 
  return redux_add(s);
}

dvec splash::operator * (const dsmatrix &M, const dvec &v) {

  //sanity checks
  if(M.N != v.N) {
    throw runtime_error("Attempt to multiply incompatible matrix and vector"); 
  }

  //create a dvec to hold the result of the matrix vector multiplication and
  //register its buffer with opencl
  dvec Mv{};
  Mv.N = M.N;
  Mv.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double)*M.N);

  //matrix vector kernel setup and invocation
  cl::Kernel kmul_mv = cl::Kernel(ocl::get().libsplash, "kmul_mv");

  kmul_mv.setArg(0, M.n);
  kmul_mv.setArg(1, M.N);
  kmul_mv.setArg(2, M.v);
  kmul_mv.setArg(3, M.ri);
  kmul_mv.setArg(4, M.ci);
  kmul_mv.setArg(5, v.v);
  kmul_mv.setArg(6, Mv.v);

  size_t gsize = M.N;
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      kmul_mv,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

  return Mv;

}

dscalar splash::ksqrt(const dscalar &s) {

  dscalar sqs{};
  sqs.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double));

  cl::Kernel ksqrt = cl::Kernel(ocl::get().libsplash, "ksqrt");

  ksqrt.setArg(0, s.v);
  ksqrt.setArg(1, 1L);
  ksqrt.setArg(2, 1L);
  ksqrt.setArg(3, sqs.v);

  ocl::get().q.enqueueNDRangeKernel(
      ksqrt,
      cl::NullRange,
      cl::NDRange{1},
      cl::NDRange{1});

  return sqs;
}

dscalar splash::knorm(const dvec &v) {

  dvec sqv{};
  sqv.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double) * v.N);
  sqv.N = v.N;

  cl::Kernel ksq = cl::Kernel(ocl::get().libsplash, "ksq");

  ksq.setArg(0, v.v);
  ksq.setArg(1, v.N);
  ksq.setArg(2, 64L);
  ksq.setArg(3, sqv.v);

  size_t gsize = v.N;
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      ksq,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});
  
  dscalar rv = redux_add(sqv);
  return ksqrt(rv);

}

dvec splash::operator / (const dvec &v, const dscalar &s) {

  dvec vds{};
  vds.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double)*v.N);
  vds.N = v.N;

  cl::Kernel kdiv_vs = cl::Kernel(ocl::get().libsplash, "kdiv_vs");

  kdiv_vs.setArg(0, v.v);
  kdiv_vs.setArg(1, s.v);
  kdiv_vs.setArg(2, v.N);
  kdiv_vs.setArg(3, 64L);
  kdiv_vs.setArg(4, vds.v);

  size_t gsize = v.N;
  gsize += 256 - (gsize % 256);

  ocl::get().q.enqueueNDRangeKernel(
      kdiv_vs,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

  return vds;
}

dvec splash::operator + (const dvec &a, const dvec &b) {

  //sanity check
  if(a.N != b.N) {
    throw runtime_error("Attempt to add incompatible vectors"); 
  }

  dvec ab{};
  ab.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double)*a.N);
  ab.N = a.N;

  cl::Kernel kadd_vv = cl::Kernel(ocl::get().libsplash, "kadd_vv");

  kadd_vv.setArg(0, a.v);
  kadd_vv.setArg(1, b.v);
  kadd_vv.setArg(2, a.N);
  kadd_vv.setArg(3, 64L);
  kadd_vv.setArg(4, ab.v);

  size_t gsize = a.N;
  gsize += 256 - (gsize & 256);

  ocl::get().q.enqueueNDRangeKernel(
      kadd_vv,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

  return ab;
}

dvec splash::operator - (const dvec &a, const dvec &b) {

  //sanity check
  if(a.N != b.N) {
    throw runtime_error("Attempt to subtract incompatible vectors"); 
  }

  dvec ab{};
  ab.v = cl::Buffer(ocl::get().ctx,
      CL_MEM_READ_WRITE,
      sizeof(double)*a.N);
  ab.N = a.N;

  cl::Kernel ksub_vv = cl::Kernel(ocl::get().libsplash, "ksub_vv");

  ksub_vv.setArg(0, a.v);
  ksub_vv.setArg(1, b.v);
  ksub_vv.setArg(2, a.N);
  ksub_vv.setArg(3, 64L);
  ksub_vv.setArg(4, ab.v);

  size_t gsize = a.N;
  gsize += 256 - (gsize & 256);

  ocl::get().q.enqueueNDRangeKernel(
      ksub_vv,
      cl::NullRange,
      cl::NDRange{gsize},
      cl::NDRange{256});

  return ab;
}


//shape ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

size_t
Shape::wgCount(size_t dim) {
  
  if(dim >= G.dimensions() || dim >= L.dimensions()) {
    throw runtime_error(
        "Workgroup count in dimension " + to_string(dim) + " requested for"
      + " workgroup of dimension " 
      + to_string(fmin(G.dimensions(), L.dimensions())));
  }
  
  return ceil(G[dim] /  (float)L[dim]);

}

size_t
Shape::wgCount() {

  size_t c{1};
  for(size_t i=0; i<G.dimensions(); ++i) { c *= wgCount(i); }
  return c; 

}
  
Shape::Shape(cl::NDRange G, cl::NDRange L) : G{G}, L{L} {}
