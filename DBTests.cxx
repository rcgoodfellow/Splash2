#include "DB.hxx"
#include <iostream>
#include <cassert>

using namespace splash;
using namespace std;

int main() {

  cout << "Performing SplashDB tests" << endl;
  string filename{"testDB.spd"};

  cout << "Saving subspace to file " << filename << endl;
  DB testDB_w{filename};
  dsubspace S = dsubspace::identity(16, 16);
  testDB_w.subspaces["S47"] = S;
  testDB_w.save();

  DB testDB_r{filename};
  testDB_r.load();
  dsubspace S_r = testDB_r.subspaces.at("S47");

  cout
    << "Read subspace from DB" << endl
    << "N=" << S_r.N << endl
    << "NA=" << S_r.NA << endl
    << "M=" << S_r.M << endl;
  
  assert(S == S_r);

  cout << "Basic Equality test passed" << endl;


  DB arnoldiDB{"ArnoldiTestResults.spd"};
  arnoldiDB.load();
  dsubspace Q = arnoldiDB.subspaces["Q"],
            H = arnoldiDB.subspaces["H"];

  cout << "Q" << endl << show_subspace(Q) << endl;
  
  cout << "H" << endl << show_subspace(H) << endl;

}
