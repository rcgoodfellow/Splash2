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
              vecops_st,
              sspaceops_st;

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

/*= DeviceElement *===========================================================*
 *
 * A DeviceElement is the base class for all mathematical objects that are
 * used in splash. Conceptually this class is really just a pointer to device
 * memory with a bit of extra data.
 *
 * This class is necessary over the basic OpenCL facilities to allow for finer
 * granularity objects than are allowed by the OpenCL runtime.
 */
struct DeviceElement {

  size_t size, offset;
  cl::Buffer memory;

  /* Creates a DeviceElement of +size @size, +offset 0 and allocates the 
   * required +memory
   */
  DeviceElement(size_t size);

  /* Creates a DeviceElement of +size @size, +offset @offset using @memory
   */
  DeviceElement(size_t size, size_t offset, cl::Buffer memory);

};

struct Vector {

};

struct dvec;

struct dscalar {
  cl::Buffer v;

  double readback();
};

struct dsubvec {
  size_t begin, end;
  cl::Buffer v;

  size_t N() const;
  size_t NA;

  dsubvec(size_t begin, size_t end, cl::Buffer v, size_t NA);

  dsubvec & operator = (const dvec &);
  dsubvec & operator = (const dsubvec &);
  dsubvec & operator = (const dscalar &);

  double* readback();

};


struct dvec {

  size_t N, NA{0};
  cl::Buffer v;
  _cl_buffer_region *br{nullptr};

  static dvec ones(size_t);

  dvec() = default;
  explicit dvec(size_t N);

  dvec & operator = (const dvec &);
  dsubvec operator() (size_t idx);
  dsubvec operator() (size_t begin, size_t end);

  double* readback() const;

};


using sm_row = std::vector<sm_sub_element>;

struct dsmatrix {
  size_t N, n;
  cl::Buffer ri, ci, v;
  SparseMatrix *M;
  std::vector<sm_row> elems;
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
  double *_data{nullptr};

  explicit dsubspace(size_t N, size_t M, bool alloc = true);
  dsubspace() = default;
  dsubspace(const dsubspace &) = default;
  dsubspace & operator = (const dsubspace&) = default;

  dvec operator () (size_t idx);
  dsubspace operator () (size_t si, size_t fi);
  void zero();
  double* readback() const;

  static dsubspace identity(size_t N, size_t M);

  bool operator == (const dsubspace &);

};

//std::string show_vec(double *v, size_t N);
std::string show_vec(const dvec &);
std::string show_matrix(double *A, size_t N, size_t M);
//std::string show_subspace(double *A, size_t N, size_t NA, size_t M);
std::string show_subspace(const dsubspace &);

dvec transmult(const dsubspace &, const dvec &);


//reducers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dscalar redux_add(const dvec&);

//operators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//vector ......................................................................
dscalar norm(const dvec&);

//scalar ......................................................................
dscalar ksqrt(const dscalar&);

//vector-vector ...............................................................
dscalar operator * (const dvec&, const dvec&);
dvec operator + (const dvec&, const dvec&);
dvec operator + (const dsubvec&, const dvec&);
dvec operator - (const dvec&, const dvec&);

//vector-scalar ...............................................................
dvec operator / (const dvec&, const dscalar&);
dvec operator / (const dvec&, const dsubvec&);

//matrix-vector ...............................................................
dvec operator * (const dsmatrix&, const dvec&);

//subspace-subvec .............................................................
dvec operator * (const dsubspace&, const dsubvec&);

//subspace-vec ................................................................
dvec operator * (const dsubspace&, const dvec&);


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dvec new_dvec(std::vector<double> values);


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
