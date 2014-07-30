#include "Splash.hxx"
#include <iostream>

using namespace std;
using namespace splash;

int main() {

  //Create two vectors on the GPU
  dvec q0 = new_dvec({1, 3, 5, 7, 9});
  dvec q1 = new_dvec({2, 4, 6, 8, 10});

  //Have the GPU perform the dotproduct and create a scalar result handle q0q1
  dscalar q0q1 = q0 * q1;
 
  //read the result of the dot product from the GPU back to the host
  double q0q1_h = q0q1.readback();
  std::cout << "Dot Product Result: " << q0q1_h << std::endl;

  //Create a sparse matrix
  dsmatrix A =  new_dsmatrix({
      {{0,4},{1,4}},
      {{0,4},{3,4},{2,3}},
      {{3,5},{2,7},{1,2}},
      {{2,5},{1,3},{4,7},{3,15}},
      {{3,7},{4,7}}
  });

  //perform matrix vector multiplication on the GPU
  dvec Aq0 = A * q0;
  double *Aq0_h = Aq0.readback();
  std::cout << "Aq0 : " << show_vec(Aq0_h, Aq0.N) << std::endl;

  //normalize a vector on the GPU
  dscalar nq0 = knorm(q0);
  std::cout << "norm(q0) : " << nq0.readback() << std::endl;

  dvec nzq0 = q0 / nq0;
  double *nzq0_h = nzq0.readback();
  std::cout << "nzq0 : " << show_vec(nzq0_h, nzq0.N) << std::endl;

  q1 = q1 / knorm(q1);
  double *q1_h = q1.readback();
  std::cout << "q1 : " << show_vec(q1_h, q1.N) << std::endl;

  dvec q2 = A * q1;
  double *q2_h = q2.readback();
  std::cout << "q2 : " << show_vec(q2_h, q2.N) << std::endl;

  dvec q12 = q1 + q2;
  double *q12_h = q12.readback();
  std::cout << "q12 : " << show_vec(q12_h, q12.N) << std::endl;
  
  dvec q21 = q2 - q1;
  double *q21_h = q21.readback();
  std::cout << "q21 : " << show_vec(q21_h, q21.N) << std::endl;

  return EXIT_SUCCESS;
}
