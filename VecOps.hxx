#ifndef SPLASH_VECOPS_HXX
#define SPLASH_VECOPS_HXX

#include "Vector.hxx"
#include "Matrix.hxx"

#include <stdexcept>

namespace splash {

Vector operator * (const Matrix &A, const Vector &x);

}

#endif
