export UNAME = $(shell uname)

MAJ_VERSION=1
MIN_VERSION=0
FUL_VERSION=$(MAJ_VERSION).$(MIN_VERSION)

CFLAGS_BASE=-Wall -fPIC -std=c++11 -Wc++11-extensions -I/opt/rocm/include -I/opt/rocm/rocrand/include/ -c

LFLAGS_LIB=-shared -Wl,-soname

EXEC_FLAGS=-std=c++11 -Wc++11-extensions -ldl -o

all: GerchbergSaxton test cleanup
	
GerchbergSaxton:
	mkdir -p Release
	hipcc $(CFLAGS_BASE) GerchbergSaxton.cpp CPU.cpp GPU.cpp
	hipcc $(LFLAGS_LIB),GerchbergSaxton.so.$(MAJ_VERSION) -o ./Release/GerchbergSaxton.so.$(FUL_VERSION) GerchbergSaxton.o CPU.o GPU.o -lfftw3 -lhiprand -lrocfft
	ln -sf GerchbergSaxton.so.$(FUL_VERSION) ./Release/GerchbergSaxton.so

test:
	mkdir -p Release
	clang++ test.cpp lodepng.cpp $(EXEC_FLAGS) ./Release/test.exe
	
cleanup:
	rm -f ./Release/*.o *.o
	
clean:
	mkdir -p Release
	rm -f ./Release/*.o ./Release/*.exe ./Release/*.so ./Release/*.so.* ./Release/*.dylib
