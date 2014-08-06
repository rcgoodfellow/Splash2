CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)

SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_FLAGS "-DMKL_ILP64 -m64 -std=c++11 -Wall -Wextra -fcolor-diagnostics -fpic -O3 -g")
SET(CMAKE_C_FLAGS "-DMKL_ILP64 -m64 -std=c11 -Wall -Wextra -fcolor-diagnostics -O3 -g")

SET(CMAKE_EXE_LINKER_FLAGS "-std=c++11")
SET(CMAKE_MODULE_LINKER_FLAGS "-std=c++11")
SET(CMAKE_SHARED_LINKER_FLAGS "-std=c++11")

ADD_LIBRARY(SplashRuntime SHARED
  Splash.cxx Splash.hxx
  KRedux.cxx KRedux.hxx
  SparseMatrix.c SparseMatrix.h
  SparseMatrix.cxx SparseMatrix.hxx
  Arnoldi.cxx Arnoldi.hxx
  VecOps.cxx VecOps.hxx
  DB.cxx DB.hxx
  )

ADD_EXECUTABLE(HelloSplash HelloSplash.cxx)
TARGET_LINK_LIBRARIES(HelloSplash SplashRuntime OpenCL)

ADD_EXECUTABLE(ArnoldiTests ArnoldiTests.cxx)
TARGET_LINK_LIBRARIES(ArnoldiTests SplashRuntime OpenCL)

ADD_EXECUTABLE(DBTests DBTests.cxx)
TARGET_LINK_LIBRARIES(DBTests SplashRuntime OpenCL)

ADD_EXECUTABLE(VectorTests VectorTests.cxx)
TARGET_LINK_LIBRARIES(VectorTests SplashRuntime OpenCL)

ADD_EXECUTABLE(MatrixTests MatrixTests.cxx)
TARGET_LINK_LIBRARIES(MatrixTests SplashRuntime OpenCL)

ADD_EXECUTABLE(SparseMatrixTests SparseMatrixTests.cxx)
TARGET_LINK_LIBRARIES(SparseMatrixTests SplashRuntime OpenCL)

ADD_EXECUTABLE(Arnoldi2 Arnoldi2.cxx)
TARGET_LINK_LIBRARIES(Arnoldi2 SplashRuntime OpenCL)

LINK_DIRECTORIES(/opt/intel/mkl/lib/intel64)

ADD_EXECUTABLE(MVDotTest MVDotTest.cxx)
TARGET_LINK_LIBRARIES(MVDotTest SplashRuntime 
  OpenCL 
  mkl_intel_ilp64 mkl_core mkl_intel_thread iomp5 dl pthread m
  )
