GCC=g++
CFLAGS=-I. -O3

S: S.cpp
	$(GCC) $(CFLAGS) S.cpp -o S

svr: svr.cpp
	$(GCC) $(CFLAGS) svr.cpp -o svr
