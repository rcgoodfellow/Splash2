#include "Splash.hxx"
#include "SMatrix.hxx"
#include "Matrix.hxx"
#include "Vector.hxx"
#include <iostream>

using namespace splash;
using namespace std;

int main() {

  cout << "Arnoldi 2 Test" << endl;

  SMatrix A(5,5,4, {
      {{0,0.4},{1,0.24}},
      {{0,0.74},{3,0.4},{2,0.3}},
      {{3,0.5},{2,0.9},{1,0.7}},
      {{2,0.5},{1,0.83},{4,0.7},{3,0.65}},
      {{3,0.7},{4,0.7}}
  });

  Vector x({0.47, 0.32, 0.34, 0.41, 0.28});

  Matrix Q{5,10}, H{5,5};
  Q.zero(); H.zero();

  Q.C(0) = x / x.norm();


  cout << "Q" << endl << Q.show() << endl;

  return EXIT_SUCCESS;
}
