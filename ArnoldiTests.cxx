#include "Arnoldi.hxx"

using namespace std;
using namespace splash;

int main() {

  dvec x0 = new_dvec({0.47, 0.32, 0.74, 0.41, 0.68});
  dsmatrix A =  new_dsmatrix({
      {{0,0.4},{1,0.4}},
      {{0,0.74},{3,0.4},{2,0.3}},
      {{3,0.5},{2,0.9},{1,0.7}},
      {{2,0.5},{1,0.83},{4,0.7},{3,0.65}},
      {{3,0.7},{4,0.7}}
  });

  Arnoldi arnoldi{A, x0};

  arnoldi();

  return EXIT_SUCCESS;

}
