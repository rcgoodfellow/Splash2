#include "VecOps.hxx"

using namespace splash;
using std::runtime_error;

#include <iostream>
using namespace std;

//need to make transposed matrix type to wrap this shit
Vector splash::operator * (const Matrix &A, const Vector &x) {

  if(A.M() != x.N()) { throw runtime_error("nonconformal operation"); }

  Vector Ax{A.N()};

  //~~~magic numbers~~~
  unsigned short 
    T{16384},                    //Total # of threads
    t{256},                     //Threads / group
    G{64},                      //# of groups
    CPG{(unsigned short)ceil(A.N()/(float)G)}, //Columns per group
    IPT{(unsigned short)ceil((A.M()*CPG)/(float)t)}; //Items per thread

  cout << "CPG:" << CPG << endl;
  cout << "IPT:" << IPT << endl;
  

  cl::Kernel k{ocl::get().libsplash, "tmvm"};
  k.setArg(0, A.memory());
  k.setArg(1, x.memory());
  k.setArg(2, (unsigned int)A.M());
  k.setArg(3, (unsigned int)A.N());
  k.setArg(4, CPG);
  k.setArg(5, IPT);
  k.setArg(6, cl::Local(t*sizeof(double)));
  k.setArg(7, Ax.memory());

  ocl::get().q.enqueueNDRangeKernel(k, 
      cl::NullRange, cl::NDRange{T}, cl::NDRange{t});

  return Ax;
}

/*
Vector splash::operator * (const Matrix &A, const Vector &x) {

  if(A.M() != x.N()) { throw runtime_error("nonconformal operation"); }

  Vector Ax{A.N()};

  cl::Kernel k{ocl::get().libsplash, "tmvm_simple"};
  k.setArg(0, A.memory());
  k.setArg(1, x.memory());
  k.setArg(2, (unsigned int)A.M());
  k.setArg(3, (unsigned int)A.N());
  k.setArg(4, Ax.memory());

  ocl::get().q.enqueueNDRangeKernel(k,
      cl::NullRange, cl::NDRange{x.N()/4}, cl::NDRange{128});

  return Ax;

}
*/
