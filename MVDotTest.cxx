
#include "Matrix.hxx"
#include "Vector.hxx"
#include "VecOps.hxx"

#include <mkl/mkl.h>

#include <iostream>
#include <chrono>
#include <random>

#define MKL_NUM_THREADS 8

using namespace splash;
using namespace std;
using namespace std::chrono;

random_device rd;
uniform_real_distribution<double> v_dst{0.5,10};
default_random_engine re{rd()};

double* buildRandomMatrix(uint32_t M, uint32_t N) {

  double *A = (double*)malloc(sizeof(double)*N*M);

  for(size_t i=0; i<M*N; ++i) {

    A[i] = rand() % 10 * drand48();

  }

  return A;
}

int main() {

  uint32_t M = 20000, N = 1000;

  cout << "Generating test data...   " << flush;
  double *A = buildRandomMatrix(M, N);
  double *b = buildRandomMatrix(M, 1);
  cout << "Done" << endl;

  double *mkl_result = (double*)malloc(sizeof(double)*N);

  cout << "mkl performance" << endl;
  auto start = high_resolution_clock::now();
  cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1, A, N, b, 1, 0, mkl_result, 1);
  auto end  = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(end-start);
  cout << dt.count()/1000.0 << " ms" << endl;

  Matrix C{M, N, A};
  Vector d{M, b};

  Vector e{N};
  for(size_t i=0; i<1; ++i) {
    e = C * d;
  }

  const double *splash_result = e.data();

  cout << "mkl:" << endl;
  for(size_t i=0; i<15; ++i) { cout << mkl_result[i] << endl; }
  cout << endl;
  
  cout << "splash:" << endl;
  for(size_t i=0; i<15; ++i) { cout << splash_result[i] << endl; }
  cout << endl;

  free(A);
  free(b);

}
