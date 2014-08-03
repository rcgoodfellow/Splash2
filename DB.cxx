#include "DB.hxx"

using namespace splash;
using std::string;
using std::ofstream;
using std::ifstream;
using std::ios_base;

#include <iostream>

DB::DB(string filename) : filename{filename} {}

void DB::load() {

  ifstream ifs{filename, ios_base::binary};

  while(ifs.peek() != EOF) {
    dsubspace S;

    size_t slen;
    ifs.read((char*)&slen, sizeof(size_t));
    char *s = (char*)malloc(slen+1);
    ifs.read(s, slen);
    s[slen] = 0;

    ifs.read((char*)&S.N, sizeof(size_t));
    ifs.read((char*)&S.NA, sizeof(size_t));
    ifs.read((char*)&S.M, sizeof(size_t));

    double *D = (double*)malloc(sizeof(double)*S.M*S.NA);

    S._data = D;
    
    for(size_t i=0; i<S.M*S.NA; ++i) { ifs.read((char*)(&D[i]), sizeof(double)); }

    S.v = cl::Buffer(ocl::get().ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(double)*S.M*S.NA,
        D);

    subspaces[string(s)] = S;
  }
  
}

void DB::save() {

  ofstream ofs{filename, ios_base::binary};

  for(const auto &p : subspaces) {

    size_t slen = p.first.length();
    ofs.write((char*)&slen, sizeof(size_t));
    ofs.write(p.first.c_str(), sizeof(char)*p.first.length());
    ofs.write((char*)&p.second.N, sizeof(size_t));
    ofs.write((char*)&p.second.NA, sizeof(size_t));
    ofs.write((char*)&p.second.M, sizeof(size_t));

    double *D = p.second.readback();
     
    for(size_t i=0; i<p.second.M*p.second.NA; ++i) { 
      ofs.write((char*)&D[i], sizeof(double));
    }

  }

  ofs.close();

}
