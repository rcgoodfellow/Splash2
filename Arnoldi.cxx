#include "Arnoldi.hxx"

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
 

  Q(0) = xi / norm(xi);  //first subspace elem is norm of residual

  size_t k;
  for(k=0; k<m-1; ++k) {

    //create next basis vector
    Q(k+1) = A * Q(k);
    //orthogonalize
    H(k)(0,k) = transmult(Q(0,k), Q(k+1));
    Q(k+1) = Q(k+1) - Q(0,k) * H(k)(0,k);  
    //reorthogonalize
    dvec s = transmult(Q(0,k), Q(k+1));
    Q(k+1) = Q(k+1) - Q(0,k) * s;
    H(k)(0,k) = H(k)(0,k) + s;
    //set Hessenberg subdiagonal
    H(k)(k+1) = norm(Q(k+1));               
    //test for subspace invariance break if so
    double *_alpha_ = H(k)(k+1).readback(); 
    if(*_alpha_ < 1e-6) {
      break;
    }
    //normalize Q(k+1)
    Q(k+1) = Q(k+1) / H(k)(k+1);            

  }


  //TODO: Should actually shrink subspace size
  //Q.fitM(k+1);
  //H.fit(k+1, k+1);
  //TODO: H is not really a subspace
  Q.M = k+1;
  H.N = k+1;
  H.M = k+1;

  return *this;

}
