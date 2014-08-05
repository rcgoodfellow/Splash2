#ifndef SPLASH_DEVICEELEMENT_HXX
#define SPLASH_DEVICEELEMENT_HXX

#include "Splash.hxx"

namespace splash {

/*= DeviceElement *===========================================================*
 *
 * A DeviceElement is the base class for all mathematical objects that are
 * used in splash. Conceptually this class is really just a pointer to device
 * memory with a bit of extra data and functionality.
 *
 * The design of this class follows the curiously recurring template pattern
 * API usage for a derived class is
 *  class Derived : DeviceElement<Derived>
 * subclasses must have a copy constructor or compilation will fail
 *
 */
template<class Derived, class T>
class DeviceElement {

  protected:
   /* 
    * This extra offset is necessary over the one provided by cl::Buffer to 
    * allow for finer granularity memory sub-references OpenCL runtime. In 
    * detail, when we want to create a subbuffer object, we must do so according 
    * to the * device base alignment constraints which are typically on the 
    * order of * 128 - 512 bytes. So if one wanted to create a subbuffer of say 
    * 3 doubles * which do not fall precisely on an alignment boundary it 
    * becomes necessary * to keep track of some sizing and offset information 
    * outside the OpenCL * runtime to determine where in that subbuffer the 3 
    * doubles actually reside
    */
    size_t _offset{0};

    //how far apart each element is
    size_t _stride{1};

    //handle to the device memory holding this object
    cl::Buffer _memory;

    //keeps track of subbuffer region info
    _cl_buffer_region _buffer_region{0,0};

    //a cache for the data on the host side
    T *_data;

  public:
   
    DeviceElement() = delete;
    /* Creates a DeviceElement of +size @size, +offset 0 and allocates the 
     * required +memory
     */
    explicit DeviceElement(size_t size)
      : _offset{0},
        _memory{ocl::get().ctx, CL_MEM_READ_WRITE, size*sizeof(T)},
        _data{(T*)malloc(size*sizeof(T))}
    {}

    /* Creates a DeviceElement of +size @size, +offset @offset using @memory
     */
    DeviceElement(size_t size, size_t offset, size_t stride, cl::Buffer memory) 
    : _stride{stride} {

      size_t base_align =
        ocl::get().gpu
                  .getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8; //bits->bytes

      size_t sub_offset = (offset*sizeof(T)) % base_align,
             cl_offset = (offset*sizeof(T)) - sub_offset;

      _offset = sub_offset / sizeof(T);
      _buffer_region.origin = cl_offset;
      _buffer_region.size = (size*sizeof(T)) + sub_offset;

      _memory = memory.createSubBuffer(
          CL_MEM_READ_WRITE, 
          CL_BUFFER_CREATE_TYPE_REGION,
          &_buffer_region);
     
      _data = (T*)malloc(dataSize()*sizeof(T));
      
    }

    //returns the size in elements of the memory region allocated for this object
    size_t allocationSize() const { 
      return _memory.getInfo<CL_MEM_SIZE>()/sizeof(T); 
    }

    size_t dataSize() const { return allocationSize() - _offset; }

    /* returns the size in elements of this object within the allocated memory 
     * region */
    size_t logicalSize() const { 
      return ceil((allocationSize() - _offset)/(float)_stride); }

    /*returns the offset in elements of this object*/
    size_t offset() const { return _offset; }

    //returns handle to the device memory holding this object
    cl::Buffer memory() const { return _memory; }

    const double* data() const { 

      ocl::get().q.enqueueReadBuffer(
          _memory,
          CL_TRUE,
          offset()*sizeof(T),
          dataSize()*sizeof(T),
          _data);

      return _data; 
    }

    //Copies this element and returns the copy
    Derived operator ! () const {

      Derived copy{*static_cast<const Derived*>(this)};
      copy._memory = cl::Buffer{ocl::get().ctx,
        CL_MEM_READ_WRITE,
        dataSize()*sizeof(T)};

      copy._offset = 0;
      copy._stride = _stride;
      copy._buffer_region = {0,0}; //{0,sz};
      copy._data = (T*)malloc(dataSize()*sizeof(T));

      ocl::get().q.enqueueCopyBuffer(
          _memory,
          copy._memory,
          offset()*sizeof(T),
          0L,
          dataSize()*sizeof(T));

      return copy;
    }


};

}

#endif
