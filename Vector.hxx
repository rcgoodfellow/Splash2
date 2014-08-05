#ifndef SPLASH_VECTOR_HXX
#define SPLASH_VECTOR_HXX

#include "DeviceElement.hxx"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "KRedux.hxx"

namespace splash {

class Vector : public DeviceElement<Vector, double> {

  enum class Type {Row, Column, SparseColumn};
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
      else if(type == Type::Column) {
        return Vector(
            _stride*end - _stride*begin + _stride,
            _stride*begin + _offset, 
            _stride, 
            _memory);
      }
      else /*Type::SparseColumn*/ {

        throw std::runtime_error{"Not Implemented"};
      }
    }

    size_t N() const { return logicalSize(); }

    std::string show() const {

      const double *d = data();

      std::stringstream ss;
      ss << std::setprecision(3) << std::fixed;
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
      else if(type == Type::Column) {

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

    Vector redux_add() {

      Vector r{1};

      cl::Kernel k{ocl::get().libsplash, "redux_add"};
      Shape shp0 = KRedux::shape(N());
      size_t lsz0 = sizeof(double) * KRedux::lmemSize(shp0);
      size_t rsz0 = sizeof(double) * shp0.wgCount();

      cl::Buffer r0{ocl::get().ctx, CL_MEM_READ_WRITE, rsz0};

      k.setArg(0, _memory);
      k.setArg(1, N());
      k.setArg(2, ocl::get().ipt);
      k.setArg(3, cl::Local(lsz0));
      k.setArg(4, r0);
      ocl::get().q.enqueueNDRangeKernel(k, cl::NullRange, shp0.G, shp0.L);

      if(shp0.wgCount() > 1) {

        Shape shp1 = KRedux::shape(shp0.wgCount());
        size_t lsz1 = sizeof(double) * KRedux::lmemSize(shp1);

        k.setArg(0, r0);
        k.setArg(1, rsz0);
        k.setArg(2, ocl::get().ipt);
        k.setArg(3, cl::Local(lsz1));
        k.setArg(4, r._memory);

        ocl::get().q.enqueueNDRangeKernel(k, cl::NullRange, shp1.G, shp1.L);
      }
      else { r._memory = r0; }

      return r;

    }

    Vector sqrt() {

      Vector s{N()};
      
      cl::Kernel k{ocl::get().libsplash, "ksqrt"};

      k.setArg(0, _memory);
      k.setArg(1, N());
      k.setArg(2, ocl::get().ipt);
      k.setArg(3, s._memory);
      ocl::get().q.enqueueNDRangeKernel(k,
          cl::NullRange,
          cl::NDRange{ocl::gsize(N())},
          cl::NDRange{ocl::lsize()});

      return s;

    }

    Vector norm() {

      Vector sq{N()};

      cl::Kernel k{ocl::get().libsplash, "ksq"};
      k.setArg(0, _memory);
      k.setArg(1, N());
      k.setArg(2, ocl::get().ipt);
      k.setArg(3, sq._memory);
      ocl::get().q.enqueueNDRangeKernel(k,
          cl::NullRange,
          cl::NDRange{ocl::gsize(N())},
          cl::NDRange{ocl::lsize()});

      Vector sq_sum = sq.redux_add();
      return sq_sum.sqrt();

    }

    Vector operator/ (const Vector &x) {
      
      if(x.N() != 1){ throw std::runtime_error("nonconformal division"); }

      Vector q{N()};

      cl::Kernel k{ocl::get().libsplash, "kdiv_vs"};
      k.setArg(0, _memory);
      k.setArg(1, x._memory);
      k.setArg(2, N());
      k.setArg(3, ocl::get().ipt);
      k.setArg(4, 0L);
      k.setArg(5, 0L);
      k.setArg(6, q._memory);
      ocl::get().q.enqueueNDRangeKernel(k,
          cl::NullRange,
          cl::NDRange{ocl::gsize(N())},
          cl::NDRange{ocl::lsize()});

      return q;

    }

};

}

#endif
