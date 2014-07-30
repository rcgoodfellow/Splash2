#include "Arnoldi.hxx"

using namespace std;
using namespace splash;

int main() {

  dvec x0 = new_dvec({1, 3, 5, 7, 9});
  dsmatrix A =  new_dsmatrix({
      {{0,4},{1,4}},
      {{0,4},{3,4},{2,3}},
      {{3,5},{2,7},{1,2}},
      {{2,5},{1,3},{4,7},{3,15}},
      {{3,7},{4,7}}
  });

  Arnoldi arnoldi{A, x0};

  arnoldi();

  return EXIT_SUCCESS;

}
