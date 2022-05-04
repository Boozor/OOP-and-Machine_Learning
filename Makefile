.PHONY: all libeasynn easynn_test

all: libeasynn easynn_test

libeasynn:
	g++ -Wall src/*.cpp -fPIC -O -g -shared -o libeasynn.so

easynn_test: libeasynn
	g++ -Wall easynn_test.cpp -g -lm -L. -Wl,-rpath=. -leasynn -o easynn_test