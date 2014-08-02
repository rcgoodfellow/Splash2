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

  size_t sz;

  for(size_t k=0; k<m-1; ++k) {

    //create the next basis vector
    Q(k+1) = A * Q(k);
    
    //create the orthogonalizing components for Q(k+1)
    H(k)(0,k) = transmult(Q(0,k), Q(k+1));

    //orthogonalize Q(k+1) w.r.t Q(0,k)

    dsubspace Q0k = Q(0,k);
    dsubvec Hks = H(k)(0,k);
    dvec QH = Q0k * Hks;
    //dsubvec Hks = Hk(0,k);

    std::cout 
      << "Q(0," << k << ")" << std::endl
      << show_subspace(Q0k.readback(), Q0k.N, Q0k.NA, Q0k.M) << std::endl;

    std::cout << "H(" << k << ")(0," << k << ")" << std::endl;
    std::cout << "Hk.N()=" << Hks.N() << std::endl;
    std::cout << show_vec(Hks.readback(), Hks.N()) << std::endl;

    std::cout 
      << "QH N=" << QH.N << std::endl
      << show_vec(QH.readback(), QH.N) << std::endl;

    Q(k+1) = Q(k+1) - Q(0,k) * H(k)(0,k);

    dvec s = transmult(Q(0,k), Q(k+1));

    Q(k+1) = Q(k+1) - Q(0,k) * s;

    H(k)(0,k) = H(k)(0,k) + s;

    H(k)(k+1) = knorm(Q(k+1));

    double *_alpha_ = H(k)(k+1).readback();
    std::cout 
      << std::setprecision(6) << std::fixed
      << "~~~{alpha}~~{" << *_alpha_ << "}~~~" 
      << std::endl;

    if(*_alpha_ < 1e-6) {
      std::cout << "convergence reached" << std::endl;
      break;
      sz = k+1;
    }

    Q(k+1) = Q(k+1) / H(k)(k+1);

    sz = k+1;

  }

 
  double *Q_h = Q.readback();
  std::cout << "Q" << std::endl;
  std::cout << show_subspace(Q_h, Q.N, Q.NA, sz) << std::endl;

  /*
  dsubspace QS47 = Q(4,7);
  double *QS47_h = QS47.readback();
  std::cout << "Q47" << std::endl;
  std::cout << show_subspace(QS47_h, QS47.N, QS47.NA, QS47.M) << std::endl;
  */

  double *H_h = H.readback();
  std::cout << "H" << std::endl;
  std::cout << show_subspace(H_h, H.N, H.NA, H.M) << std::endl;

  return *this;
}
