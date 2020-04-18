export UNAME = $(shell uname)

MAJ_VERSION=1
MIN_VERSION=0
FUL_VERSION=$(MAJ_VERSION).$(MIN_VERSION)

CFLAGS_BASE=-Wall -fPIC -std=c++11 -Wc++11-extensions -I/opt/rocm/include -c

LFLAGS_LIB=-shared -Wl,-soname

EXEC_FLAGS=-std=c++11 -Wc++11-extensions -ldl -o

all: LensPhase test cleanup
	
LensPhase:
	mkdir -p Release
	hipcc $(CFLAGS_BASE) LensPhase.cpp CPU.cpp GPU.cpp
	hipcc $(LFLAGS_LIB),LensPhase.so.$(MAJ_VERSION) -o ./Release/LensPhase.so.$(FUL_VERSION) LensPhase.o CPU.o GPU.o
	ln -sf LensPhase.so.$(FUL_VERSION) ./Release/LensPhase.so

test:
	mkdir -p Release
	clang++ test.cpp ../Includes/lodepng.cpp $(EXEC_FLAGS) ./Release/test.exe
	
cleanup:
	rm -f ./Release/*.o *.o
	
clean:
	mkdir -p Release
	rm -f ./Release/*.o ./Release/*.exe ./Release/*.so ./Release/*.so.* ./Release/*.dylib
