#ifndef SPLASH_DB_HXX
#define SPLASH_DB_HXX

#include "Splash.hxx"
#include <unordered_map>
#include <string>
#include <fstream>

namespace splash {

struct DB {

  std::string filename;
  std::unordered_map<std::string, dsubspace> subspaces;

  DB(std::string filename);
  void load();
  void save();

};

}

#endif
