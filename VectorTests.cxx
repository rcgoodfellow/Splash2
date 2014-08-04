#include "Vector.hxx"
#include <iostream>
#include <cassert>

using namespace std;
using namespace splash;

int main() {

  Vector v{{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}};
  cout << "v: " << v.show() << endl;

  Vector mask{{0,0,0,0}};
  cout << "mask: " << mask.show() << endl;

  Vector a =  v(4,7),
         b = !v(8,11);
  cout << "a: " << a.show() << endl;
  cout << "b: " << b.show() << endl;

  cout << "Applying mask" << endl;
  a = mask;
  b = mask;
  
  cout << "a: " << a.show() << endl;
  cout << "b: " << b.show() << endl;
  cout << "v: " << v.show() << endl;
  
  Vector expected{{1,2,3,4,0,0,0,0,9,10,11,12,13,14,15,16}};

  assert(v == expected);

  v(4,7) = Vector{{5,6,7,8}};
  cout << "v: " << v.show() << endl;

}
