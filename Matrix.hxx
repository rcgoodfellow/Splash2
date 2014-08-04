#ifndef SPLASH_MATRIX_HXX
#define SPLASH_MATRIX_HXX

#include "DeviceElement.hxx"
#include "Vector.hxx"
#include <stdexcept>
#include <sstream>

namespace splash {

class Matrix : public DeviceElement<Matrix, double> {

  size_t _M, _N;

  public: 
    Matrix(size_t rows, size_t cols)
      : DeviceElement<Matrix, double>(rows*cols),
        _M{rows}, _N{cols} { }

    Matrix(size_t rows, size_t cols, 
           size_t row_offset, size_t col_offset, 
           cl::Buffer memory)
      : DeviceElement<Matrix, double>{rows*cols, 
                                      cols*row_offset + col_offset, 1L,
                                      memory},
        _M{rows}, _N{cols} { }

    Matrix(size_t rows, size_t cols, std::vector<double> x)
      : DeviceElement<Matrix, double>(rows*cols),
        _M{rows}, _N{cols}
    { 
      if(rows*cols != x.size()) {
        throw std::runtime_error(
            "Attempt to create matrix of size " + std::to_string(rows*cols)
            + " with a data vector of size " + std::to_string(x.size()));
      }
      ocl::get().q.enqueueWriteBuffer(
        _memory,
        CL_TRUE,
        0,
        allocationSize()*sizeof(double),
        x.data());
    }

    Vector R(size_t idx) {
      return Vector(_N, _N*idx, 1L, _memory);
    }

    Vector C(size_t idx) {
      return Vector(_M*_N - idx, idx, _N, _memory);
    }

    size_t M() const { return _M; }
    size_t N() const { return _N; }

    std::string show() const {

      const double *A = data();

      std::stringstream ss;
      ss << std::setprecision(3) << std::fixed;
      for(size_t i=0; i<_N; ++i) { 
        for(size_t j=0; j<_M; ++j) {
          ss << A[_M*i+j] << "\t";
        }
        ss << std::endl;
      }

      return ss.str();

    }

};

}

#endif
