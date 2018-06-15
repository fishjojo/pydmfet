icpc -fPIC -c c_wannier90_interface.cpp
ifort -cxxlib -nofor-main -shared -o libwannier90.so c_wannier90_interface.o -L/tigress/xingz/wannier90-2.1.0 -lwannier -lstdc++

icpc -shared -fPIC -O2 -o libwannier90.so c_wannier90_interface.cpp -L/tigress/xingz/wannier90-2.1.0 -lwannier -lstdc++

icpc -shared -fPIC -O2 -qopenmp -o libmisc.so comput_amn.cpp
