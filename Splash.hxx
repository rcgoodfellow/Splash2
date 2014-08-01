#ifndef SPLASH_HXX
#define SPLASH_HXX

#include "SparseMatrix.hxx"
  
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

#define SPLASHDIR "/home/ry/Splash2"

namespace splash {

std::string read_file(std::string filename);

struct LibSplash {

  cl::Program::Sources src;
  std::string splashdir, build_opts;

  //source text strings
  std::string redux_st,
              elemental_st,
              mvmul_st,
              mxops_st,
              vecops_st;

  LibSplash(std::string splashdir);

  cl::Program get(cl::Context ctx);

  private:
    void readSource();

};

class ocl {
  public:
    static ocl& get() {
      static ocl instance;
      return instance;
    }

    cl::Platform platform;
    cl::Context ctx;
    cl::Device gpu;
    cl::CommandQueue q;

    LibSplash libsplash_loader{SPLASHDIR};
    cl::Program libsplash;

    size_t ipt{64};
    static size_t gsize(size_t N, bool distribute_over_ipt=true) {

      float divisor = 1.0f;
      if(distribute_over_ipt) { divisor = 64.0f; }
      size_t gsz = static_cast<size_t>(ceil(N/divisor));
      gsz += 256 - (gsz % 256);
      return gsz;

    }

    static size_t lsize() { return 256; }


  private:
    ocl() {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      platform = platforms[0];

      std::vector<cl::Device> gpus;
      platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
      gpu = gpus.back();

      ctx = cl::Context(gpu);
      libsplash = libsplash_loader.get(ctx);

      q = cl::CommandQueue(ctx);
      
    }
    ocl(const ocl &) = delete;
    void operator=(const ocl &) = delete;
};

struct dvec;

struct dsubvec {
  size_t begin, end;
  dvec *parent;

  size_t N() const;

  dsubvec(size_t begin, size_t end, dvec *parent);

  dsubvec & operator = (const dvec &);
  dsubvec & operator = (const dsubvec &);

};

struct dvec {

  size_t N, NA;
  cl::Buffer v;
  _cl_buffer_region *br{nullptr};

  static dvec ones(size_t);

  dvec() = default;
  explicit dvec(size_t N);

  dvec & operator = (const dvec &);
  dsubvec operator() (size_t begin, size_t end);

  double* readback();

};

std::string show_vec(double *v, size_t N);
std::string show_matrix(double *A, size_t N, size_t M);
std::string show_subspace(double *A, size_t N, size_t NA, size_t M);


struct dsmatrix {
  size_t N, n;
  cl::Buffer ri, ci, v;
};

struct dmatrix {

  struct dcol {
    dmatrix *parent;
    size_t idx;

    dcol(dmatrix *parent, size_t idx);
    dcol& operator = (const dvec&);
  };

  size_t N, M;
  cl::Buffer v;

  dmatrix(size_t N, size_t M);

  double *readback();
  dcol C(size_t idx);
  void zero();
};

struct dsubspace {

  size_t N,   //the logical size of N
         NA,  //the actual size of N (for alignment and sub-buffering purposes
         M;   //size of the subspace
  cl::Buffer v;
  _cl_buffer_region *br{nullptr};

  dsubspace(size_t N, size_t M, bool alloc = true);
  dvec operator () (size_t idx);
  dsubspace operator () (size_t si, size_t fi);
  void zero();
  double* readback();

};

dvec transmult(const dsubspace &, const dvec &);

struct dscalar {
  cl::Buffer v;
  double readback();
};

//reducers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dscalar redux_add(const dvec&);

//operators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//vector ......................................................................
dscalar knorm(const dvec&);

//scalar ......................................................................
dscalar ksqrt(const dscalar&);

//vector-vector ...............................................................
dscalar operator * (const dvec&, const dvec&);
dvec operator + (const dvec&, const dvec&);
dvec operator - (const dvec&, const dvec&);

//vector-scalar ...............................................................
dvec operator / (const dvec&, const dscalar&);

//matrix-vector ...............................................................
dvec operator * (const dsmatrix&, const dvec&);

//subspace-subvec .............................................................
dvec operator * (const dsubspace&, const dsubvec&);


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dvec new_dvec(std::vector<double> values);

using sm_row = std::vector<sm_sub_element>;

dsmatrix new_dsmatrix(std::vector<sm_row>);

struct Shape {

  cl::NDRange G, L;

  Shape() = default;
  Shape(cl::NDRange G, cl::NDRange L);

  size_t wgCount(size_t dim);
  size_t wgCount();

};

}

#endif
