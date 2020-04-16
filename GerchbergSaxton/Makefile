export UNAME = $(shell uname)

MAJ_VERSION=1
MIN_VERSION=0
FUL_VERSION=$(MAJ_VERSION).$(MIN_VERSION)

CFLAGS_BASE=-Wall -fPIC -std=c++11 -Wc++11-extensions -I/opt/rocm/include -I/opt/rocm/rocrand/include/ -c

LFLAGS_LIB=-shared -Wl,-soname

EXEC_FLAGS=-std=c++11 -rdynamic -ldl -Wc++11-extensions -o

all: GerchbergSaxton
	
GerchbergSaxton:
	hipcc $(CFLAGS_BASE) GerchbergSaxton.cpp CPU.cpp GPU.cpp
	
	clang++ $(LFLAGS_LIB),GerchbergSaxton.so.$(MAJ_VERSION) -o GerchbergSaxton.so.$(FUL_VERSION) GerchbergSaxton.o -lfftw3
	ln -sf GerchbergSaxton.so.$(FUL_VERSION) GerchbergSaxton.so

cleanup:
	rm -f *.o
	
install: all cleanup
	
clean:
	rm -f *.o *.exe *.so *.so.* *.dylib
