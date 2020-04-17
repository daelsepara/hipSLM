#define _USE_MATH_DEFINES
#include "../Includes/lodepng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI

#undef min
#undef max

// Allocate double array on host
double* Double(int Length, double val)
{
	auto host = (double*)malloc(Length * sizeof(double));

	if (host != NULL)
	{
		for (auto x = 0; x < Length; x++)
		{
			host[x] = val;
		}
	}

	return host;
}

void phasepng(const char* filename, double* data, int gray, int xdim, int ydim)
{
	unsigned char* buffer = (unsigned char*)malloc(xdim * ydim);

	for (int index = 0; index < xdim * ydim; index++)
	{
		unsigned char c = (unsigned char)((double)gray * (data[index]) / (2.0 * M_PI));

		buffer[index] = c;
	}

	unsigned error = lodepng_encode_file(filename, buffer, xdim, ydim, LCT_GREY, 8);

	if (error)
	{
		fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
	}

	free(buffer);
}

inline double Mod(double a, double m)
{
	return a - m * floor(a / m);
}

void ParseInt(std::string arg, const char* str, const char* var, int& dst)
{
	auto len = strlen(str);

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = stoi(arg.substr(len));
				fprintf(stderr, "... %s = %d\n", var, val);
				dst = val;
			}
			catch (const std::invalid_argument& ia)
			{
				fprintf(stderr, "... %s = NaN %s\n", var, ia.what());
				exit(1);
			}
		}
	}
}

void ParseDouble(std::string arg, const char* str, const char* var, double& dst)
{
	auto len = strlen(str);

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = stod(arg.substr(len));
				fprintf(stderr, "... %s = %lf\n", var, val);
				dst = val;
			}
			catch (const std::invalid_argument& ia)
			{
				fprintf(stderr, "... %s = NaN %s\n", var, ia.what());
				exit(1);
			}
		}
	}
}

int main(int argc, char** argv)
{
	auto M = 800;  // SLM width in # of pixels
	auto N = 600;  // SLM height in # of pixels
	auto gpu = false;
	auto MirrorX = 100.0;
	auto MirrorY = 100.0;
	
	int block = 16;

	for (int i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

		ParseInt(arg, "/m=", "SLM Width", M);
		ParseInt(arg, "/n=", "SLM Height", N);
		ParseDouble(arg, "/x=", "Mirror X Displacement", MirrorX);
		ParseDouble(arg, "/y=", "Mirror Y Displacement", MirrorY);
		
		ParseInt(arg, "/blocksize=", "block size", block);
		
		if (!arg.compare("/gpu"))
		{
			gpu = true;
			
			fprintf(stderr, "... use GPU\n");
		}
	}

	char Libname[200];
	
	void *lib_handle;

	void (*Calculate)(int, void**);
	
	char *error;
	
	sprintf(Libname, "./PrismPhase.so");
	
	lib_handle = dlopen(Libname, RTLD_LAZY);

	if (!lib_handle)
	{
	  fprintf(stderr, "%s\n", dlerror());
	  exit(1);
	}
	
	Calculate = (void (*)(int, void**)) dlsym(lib_handle, "Calculate");
	if ((error = dlerror()) != NULL) 
	{
	  fprintf(stderr, "%s\n", error);
	  exit(1);
	}
	
	double *PrismPhase = Double(M * N, 0.0);
	
	// Prism Phase
	void* Params[] = { PrismPhase, &M, &N, &MirrorX, &MirrorY, &gpu };
	(*Calculate)(6, Params);
	
	phasepng("phase-mirror.png", PrismPhase, 255, M, N);
	
	free(PrismPhase);
	
	return 0;
}
