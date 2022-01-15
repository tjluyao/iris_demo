%module cords
%{
#include "cords.h"
%}
%include "std_vector.i"
%include "std_string.i"
namespace std {
  %template(VecInt) vector<int>;
}

%include "cords.h"