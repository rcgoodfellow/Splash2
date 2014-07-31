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
    static size_t gsize(size_t N) {

      size_t gsz = static_cast<size_t>(ceil(N/ocl::get().ipt));
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

struct dvec {

  size_t N;
  cl::Buffer v;
  _cl_buffer_region *br{nullptr};

  dvec & operator = (const dvec &);

  double* readback();

};

std::string show_vec(double *v, size_t N);
std::string show_matrix(double *A, size_t N, size_t M);
std::string show_subspace(double *A, size_t N, size_t M);


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

  size_t N, M;
  cl::Buffer v;

  dsubspace(size_t N, size_t M);
  dvec operator () (size_t idx);
  void zero();
  double* readback();

};

struct dscalar {
  cl::Buffer v;
  double readback();
};

//operators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//vector unary ................................................................
dscalar knorm(const dvec&);

//vector reducers .............................................................
dscalar redux_add(const dvec&);

//vector-vector binary ........................................................
dscalar operator * (const dvec&, const dvec&);
dvec operator + (const dvec&, const dvec&);
dvec operator - (const dvec&, const dvec&);

//vector-scalar binary ........................................................
dvec operator / (const dvec&, const dscalar&);

//matrix-vector binary ........................................................
dvec operator * (const dsmatrix&, const dvec&);

//scalar unary ................................................................
dscalar ksqrt(const dscalar&);

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
