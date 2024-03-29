#include "Matrix.hxx"
#include <iostream>
#include <cassert>

using namespace std;
using namespace splash;

int main() {

  Matrix M{8,8,
    { 1,0,0,0,0,0,0,0,
      0,1,0,0,0,0,0,0,
      0,0,1,0,0,0,0,0,
      0,0,0,1,0,0,0,0,
      0,0,0,0,1,0,0,0,
      0,0,0,0,0,1,0,0,
      0,0,0,0,0,0,1,0,
      0,0,0,0,0,0,0,1 }
  };

  cout << M.show() << endl;

  Vector r4 = M.R(4);
  cout << r4.show() << endl;
  Vector r4_expected{{0,0,0,0,1,0,0,0}};
  assert(r4 == r4_expected);

  r4(2,5) = {{4,2,2,4}};
  cout << M.show() << endl;

  Vector r2 = !M.R(2);
  r2 = {{2,2,2,2,2,2,2,2}};
  cout << M.show() << endl;

  Vector c2 = M.C(2);
  cout << "c2: " << c2.show() << endl;

  Vector c224 = !c2(2,4);
  cout << "c224: " << c224.show() << endl;
  c224 = {{3,3,3}};
  cout << M.show() << endl;
}
