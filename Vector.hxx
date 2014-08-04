#ifndef SPLASH_VECTOR_HXX
#define SPLASH_VECTOR_HXX

#include "DeviceElement.hxx"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace splash {

class Vector : public DeviceElement<Vector, double> {

  enum class Type {Row, Column};
  Type type{Type::Row};

  public:
    Vector(size_t size)
      : DeviceElement<Vector, double>(size) { }

    Vector(size_t size, size_t offset, size_t stride, cl::Buffer memory)
      : DeviceElement<Vector, double>{size, offset, stride, memory} { 
     
        if(stride != 1) { type = Type::Column; }
      }

    Vector(std::vector<double> x)
      : DeviceElement<Vector, double>{x.size()}
    {
      ocl::get().q.enqueueWriteBuffer(
          _memory,
          CL_TRUE,
          0L,
          allocationSize()*sizeof(double),
          x.data());
    }

    Vector operator()(size_t begin, size_t end) { 
      if(type == Type::Row) {
        return Vector( end - begin + 1L, begin, 1L, _memory); 
      }
      else {
        return Vector(
            _stride*end - _stride*begin + _stride,
            _stride*begin + _offset, 
            _stride, 
            _memory);
      }
    }

    size_t N() const { return logicalSize(); }

    std::string show() const {

      const double *d = data();

      std::stringstream ss;
      ss << "[";
      if(N() == 0) {
        ss << "]";
        return ss.str();
      }
      for(size_t i=0; i<N()-1; ++i) {
        ss << d[i*_stride] << ","; 
      }
      ss << d[(N()-1)*_stride] << "]";
      return ss.str();


    }
    
    Vector & operator = (const Vector &other) {

      if(N() != other.N()) {
        throw std::runtime_error("Attemtp to assign nonconformal vectors");
      }

      free(_data);
      _data = other._data;
     
      if(type == Type::Row) {
        ocl::get().q.enqueueCopyBuffer(
            other._memory,
            _memory,
            other._offset*sizeof(double),
            _offset*sizeof(double),
            other.logicalSize()*sizeof(double));
      }
      else {

        cl::Kernel k{ocl::get().libsplash, "vx_strset"};
        k.setArg(0, _memory);
        k.setArg(1, other._memory);
        k.setArg(2, N());
        k.setArg(3, _stride);
        k.setArg(4, _offset);
        k.setArg(5, ocl::get().ipt);

        ocl::get().q.enqueueNDRangeKernel(
            k,
            cl::NullRange,
            cl::NDRange{ocl::gsize(N())},
            cl::NDRange{ocl::lsize()});

      }
    
      return *this;
    }

    bool operator == (const Vector &other) {

      if(N() != other.N()) { return false; }

      cl::Kernel k{ocl::get().libsplash, "vx_eq"};

      unsigned long diff_h{0};
      cl::Buffer diff{ocl::get().ctx,
        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(unsigned long),
        &diff_h};

      k.setArg(0, _memory);
      k.setArg(1, other._memory);
      k.setArg(2, N());
      k.setArg(3, ocl::get().ipt);
      k.setArg(4, 1e-6);
      k.setArg(5, diff);

      ocl::get().q.enqueueNDRangeKernel(
          k,
          cl::NullRange,
          cl::NDRange{ocl::gsize(N())},
          cl::NDRange{ocl::lsize()});

      ocl::get().q.enqueueReadBuffer(
          diff,
          CL_TRUE,
          0,
          sizeof(unsigned long),
          &diff_h);

      return diff_h == 0L;

    }

};

}

#endif
