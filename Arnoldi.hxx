#ifndef SPLASH_ARNOLDI_HXX
#define SPLASH_ARNOLDI_HXX

#include "Splash.hxx"

namespace splash {

struct Arnoldi {

  dsmatrix A; 
  dsubspace Q, H;
  dvec xi;
  size_t m{10};

  Arnoldi(dsmatrix A, dvec xi);

  Arnoldi & operator()();

};

}

#endif
