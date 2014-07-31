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

  //zero out for debugging
  Q.zero();
  H.zero();

  Q(0) = xi / knorm(xi);

  for(size_t k=0; k<m-1; ++k) {

    //fascinating that this converges
    Q(k+1) = (A * Q(k)) / knorm(Q(k));

    //Q(k+1) = A * Q(k);

    
    //H(k) = transmult(Q(0,k), H(k));

  }

  
  double *Q_h = Q.readback();
  std::cout << "Q" << std::endl;
  std::cout << show_subspace(Q_h, Q.N, Q.NA, Q.M) << std::endl;

  dsubspace QS47 = Q(4,7);
  double *QS47_h = QS47.readback();
  std::cout << "Q47" << std::endl;
  std::cout << show_subspace(QS47_h, QS47.N, QS47.NA, QS47.M) << std::endl;

  return *this;
}
