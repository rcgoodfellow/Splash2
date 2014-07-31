#include "Arnoldi.hxx"
#include <iostream>

using namespace splash;
using std::runtime_error;

Arnoldi::Arnoldi(dsmatrix A, dvec xi) : A(A), Q{A.N, 10}, H{10,10}, xi(xi) {

  if(A.N != xi.N) {
    throw runtime_error("Arnoldi arguments of incompatible size"); 
  }

}

Arnoldi& Arnoldi::operator()() {

  Q.zero();
  Q(0) = xi / knorm(xi);
  double *Q_h = Q.readback();
  std::cout << show_subspace(Q_h, Q.N, Q.M) << std::endl;

  for(size_t k=0; k<m; ++k) {


  }

  return *this;
}
