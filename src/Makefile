compile:
	swig -c++ -python cords.i
	g++ -c -O3 -fPIC -std=c++11 -fopenmp cords.cpp cords_wrap.cxx -I/anaconda3/include/python3.7m
	g++ -shared -O3 -std=c++11 -fopenmp -Wl,-soname,_cords.so -o _cords.so cords.o cords_wrap.o
	swig -c++ -python shrink_cpp.i
	g++ -c -O3 -fPIC -std=c++11 -fopenmp shrink_cpp.cpp shrink_cpp_wrap.cxx -I/anaconda3/include/python3.7m
	g++ -shared -O3 -std=c++11 -fopenmp -Wl,-soname,_shrink_cpp.so -o _shrink_cpp.so shrink_cpp.o shrink_cpp_wrap.o

clean:
	rm *.o *.so *.cxx cords.py shrink_cpp.py