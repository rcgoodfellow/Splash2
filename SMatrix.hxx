#ifndef SPLASH_SMATRIX_HXX
#define SPLASH_SMATRIX_HXX

#include "SparseMatrix.h"
#include "Vector.hxx"
#include <stdexcept>
#include <sstream>
#include <vector>

namespace splash {

struct RowElement { uint16_t idx; double value; };
using Row = std::vector<RowElement>;

//only supporting square sparse matricies at this time
class SMatrix {

  uint32_t _M, //# of rows
           _N; //# of columns
  uint16_t _n; //max row size

  double *_V; //values
  cl::Buffer _V_;

  uint32_t *_C; //column indicies
  cl::Buffer _C_;

  uint16_t *_R; //row sizes
  cl::Buffer _R_;

  public:
    explicit
    SMatrix(uint32_t M, uint32_t N, uint16_t n, std::vector<Row> rows)
    : _M{M}, _N{N}, _n{n} {

      _V = (double*)malloc(sizeof(double)*_M*_n);
      _C = (uint32_t*)malloc(sizeof(uint32_t)*_M*_n);
      _R = (uint16_t*)malloc(sizeof(uint16_t)*_M);

      for(uint32_t i=0; i<_M; ++i) {

        _R[i] = rows[i].size();

        for(uint16_t j=0; j<rows[i].size(); ++j) {
          
          RowElement &e = rows[i][j];
          _C[i*_n+j] = e.idx;
          _V[i*_n+j] = e.value;

        }}

      _V_ = cl::Buffer(ocl::get().ctx,
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(double)*_M*_n,
          _V);

      _C_ = cl::Buffer(ocl::get().ctx,
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(uint32_t)*_M*_n,
          _C);

      _R_ = cl::Buffer(ocl::get().ctx,
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(uint16_t)*_M,
          _R);

    }

    Vector operator* (const Vector &x) {

      if(_N != x.N()){ 
        throw std::runtime_error("nonconformal multiplication"); }

      Vector Ax{_M};

      cl::Kernel k{ocl::get().libsplash, "smv_mul"};
      k.setArg(0, _M);
      k.setArg(1, _N);
      k.setArg(2, _n);
      k.setArg(3, _V_);
      k.setArg(4, _C_);
      k.setArg(5, _R_);
      k.setArg(6, x.memory());
      k.setArg(7, Ax.memory());

      ocl::get().q.enqueueNDRangeKernel(
          k,
          cl::NullRange,
          cl::NDRange{ocl::gsize(_M,false)},
          cl::NDRange{ocl::lsize()});

      return Ax;

    }

    std::string show() {

      std::stringstream ss;
      ss << std::setprecision(3) << std::fixed;

      ss << "M:" << _M << " N:" << _N << " n:" << _n << std::endl;
      std::string hdr = ss.str();
      ss.str("");

      ss << "v::[";
      for(uint32_t i=0; i<_M; ++i) {
        ss << color(i);
        for(uint16_t j=0; j<_R[i]; ++j) {
          ss << _V[_n*i+j] << " ";
        }}
      std::string vstr = ss.str();
      ss.str("");
      vstr.pop_back();
      vstr += "\e[0m]\n";
      
      ss << "c::[";
      for(uint32_t i=0; i<_M; ++i) {
        ss << color(i);
        for(uint16_t j=0; j<_R[i]; ++j) {
          ss << _C[_n*i+j] << " ";
        }}
      std::string cstr = ss.str();
      ss.str("");
      cstr.pop_back();
      cstr +=  "\e[0m]\n";

      ss << "r::[";
      for(uint32_t i=0; i<_M; ++i) { ss << color(i) << _R[i] << " "; }
      std::string rstr = ss.str();
      ss.str("");
      rstr.pop_back();
      rstr += "\e[0m]\n";

      return hdr + vstr + cstr + rstr;

    }

  private:
    std::string color(size_t i) {

      if(i%2 == 0) { return "\e[0;32m"; }
      else { return "\e[0;34m"; }

    }
};

}

#endif
