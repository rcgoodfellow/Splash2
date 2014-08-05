#include "SMatrix.hxx"
#include "Vector.hxx"
#include <iostream>

using namespace splash;
using namespace std;

int main() {

  cout << "Conducting Sparse Matrix Tests" << endl;

  SMatrix M{8,8,3,
    { {{0,1}}, 
      {{1,1}}, 
      {{2,1}}, 
      {{2,2},{3,1},{6,1}}, 
      {{3,1},{4,1},{5,2}}, 
      {{5,1}}, 
      {{6,1}}, 
      {{7,1}}  }};

  cout << M.show() << endl;

  Vector v{{4,7,4,7,4,7,4,7}};

  cout << "v::" << v.show() << endl;
  
  Vector Mv = M * v;

  cout << "Mv::" << Mv.show() << endl;

  return EXIT_SUCCESS;

}
