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
  cout << "Dot Product Result: " << q0q1_h << endl;

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
  cout << "Aq0 : " << show_vec(Aq0) << endl;

  //normalize a vector on the GPU
  dscalar nq0 = norm(q0);
  cout << "norm(q0) : " << nq0.readback() << endl;

  dvec nzq0 = q0 / nq0;
  cout << "nzq0 : " << show_vec(nzq0) << endl;

  q1 = q1 / norm(q1);
  cout << "q1 : " << show_vec(q1) << endl;

  dvec q2 = A * q1;
  cout << "q2 : " << show_vec(q2) << endl;

  dvec q12 = q1 + q2;
  cout << "q12 : " << show_vec(q12) << endl;
  
  dvec q21 = q2 - q1;
  cout << "q21 : " << show_vec(q21) << endl;

  return EXIT_SUCCESS;
}
