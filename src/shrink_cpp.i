%module shrink_cpp
%{
#include "shrink_cpp.h"
%}
%include "std_vector.i"
namespace std {
  %template(VecInt) vector<int>;
  %template(VecFloat) vector<float>;
  %template(VecVecFloat) vector< vector<float> >;
  %template(VecVecInt) vector< vector<int> >;
}

%include "shrink_cpp.h"