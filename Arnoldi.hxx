#ifndef SPLASH_ARNOLDI_HXX
#define SPLASH_ARNOLDI_HXX

#include "Splash.hxx"

namespace splash {

struct Arnoldi {

  dsmatrix A; 
  dmatrix Q, H;
  dvec q0;

  Arnoldi(dsmatrix A, dvec q0);

  Arnoldi & operator()();

};

}

#endif
