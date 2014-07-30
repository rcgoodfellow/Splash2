#include "Arnoldi.hxx"
#include <iostream>

using namespace splash;
using std::runtime_error;

Arnoldi::Arnoldi(dsmatrix A, dvec q0) : A(A), Q(A.N, 10), H(10, 10), q0(q0) {

  if(A.N != q0.N) {
    throw runtime_error("Arnoldi arguments of incompatible size"); 
  }

}

Arnoldi& Arnoldi::operator()() {

  Q.zero();
  double *Q_h = Q.readback();
  std::cout << show_matrix(Q_h, Q.N, Q.M) << std::endl;
  
  Q.C(0) = q0 / knorm(q0);
  Q_h = Q.readback();
  std::cout << show_matrix(Q_h, Q.N, Q.M) << std::endl;

  return *this;
}
