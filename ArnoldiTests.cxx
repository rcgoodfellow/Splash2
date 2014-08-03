#include "Arnoldi.hxx"
#include "DB.hxx"
#include <iostream>

using namespace std;
using namespace splash;

int main() {

  dvec b = new_dvec({0.47, 0.32, 0.34, 0.41, 0.28});
  dsmatrix A =  new_dsmatrix({
      {{0,0.4},{1,0.24}},
      {{0,0.74},{3,0.4},{2,0.3}},
      {{3,0.5},{2,0.9},{1,0.7}},
      {{2,0.5},{1,0.83},{4,0.7},{3,0.65}},
      {{3,0.7},{4,0.7}}
  });

  cout << "b: " << show_vec(b) << endl << endl;

  //implicitly guessing x to be the zero vector
  Arnoldi arnoldi{A, b};
  arnoldi();
  
  dvec Qb = transmult(arnoldi.Q, arnoldi.xi);

  //Display Results
  cout << "Q" << endl;
  cout << show_subspace(arnoldi.Q) << endl;

  cout << "Qb" << endl;
  cout << show_vec(Qb) << endl << endl;

  cout << "H" << endl;
  cout << show_subspace(arnoldi.H) << endl;

  /*
  //Save Results to a Splash Data File
  DB db{"ArnoldiTestResults.spd"};
  db.subspaces["Q"] = arnoldi.Q;
  db.subspaces["H"] = arnoldi.H;
  db.save();
  */

  return EXIT_SUCCESS;

}
