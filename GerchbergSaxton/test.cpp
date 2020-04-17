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

void amppng(const char* filename, double* data, int xdim, int ydim, double scale = 1.0)
{
	unsigned char* buffer = (unsigned char*)malloc(xdim * ydim);

	for (int index = 0; index < xdim * ydim; index++)
	{
		unsigned char c = (unsigned char)(data[index] * scale);

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

double* loadpng(const char* filename, int& xdim, int& ydim, double scale = 1.0)
{
	unsigned error;
	unsigned char* image;
	unsigned w, h;

	// load PNG
	error = lodepng_decode24_file(&image, &w, &h, filename);

	// exit on error
	if (error)
	{
		fprintf(stderr, "decoder error %u: %s\n", error, lodepng_error_text(error));
		exit(1);
	}

	// allocate w x h double image 
	int memsize = w * h * sizeof(double);

	auto png = (double*)malloc(memsize);

	if (!png)
	{
		fprintf(stderr, "unable to allocate %u bytes for image\n", memsize);
		exit(1);
	}

	for (int index = 0; index < (int)(w * h); index++)
	{
		unsigned char r, g, b;

		r = image[3 * index + 0]; // red
		g = image[3 * index + 1]; // green
		b = image[3 * index + 2]; // blue

		// convert RGB to grey
		auto val = (0.299 * (double)r + 0.587 * (double)g + 0.114 * (double)b) * scale;

		png[index] = val;
	}

	xdim = w;
	ydim = h;

	free(image);

	return png;
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

int main(int argc, char** argv)
{
	auto M = 800;  // SLM width in # of pixels
	auto N = 600;  // SLM height in # of pixels
	auto Ngs = 1000;  // Ngs
	auto gpu = false;
	
	char InputFile[200];
	InputFile[0] = '\0';

	int block = 16;

	for (int i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

		ParseInt(arg, "/m=", "SLM Width", M);
		ParseInt(arg, "/n=", "SLM Height", N);
		ParseInt(arg, "/i=", "GS Iterations", Ngs);
		
		if (!arg.compare(0, 8, "/target=") && arg.length() > 8)
		{
			strncpy(InputFile, &argv[i][8], sizeof(InputFile));
		}

		ParseInt(arg, "/blocksize=", "block size", block);
		
		if (!arg.compare("/gpu"))
		{
			gpu = true;
			
			fprintf(stderr, "... use GPU\n");
		}
	}
	
	if (strlen(InputFile) > 0)
	{
		int targetw;
		int targeth;

		double* target = loadpng(InputFile, targetw, targeth, 1.0);

		double h = 20e-6;  // hologram pixel size

		bool gaussian = false;
		double r = 960e-6;  // input gaussian beam waist

		// No aperture
		int aperture = 0;
		int aperturew = targetw;
		int apertureh = targeth;

		char Libname[200];
		
		void *lib_handle;

		void (*Calculate)(int, void**);
		
		char *error;
		
		sprintf(Libname, "./GerchbergSaxton.so");
		
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
		
		double *GerchbergSaxtonPhase = Double(M * N, 0.0);
		
		// Gerchberg-Saxton
		void* Params[] = { GerchbergSaxtonPhase, &M, &N, &Ngs, &h, &gaussian, &r, &aperture, &aperturew, &apertureh, target, &targetw, &targeth, &gpu };
		(*Calculate)(14, Params);
		
		phasepng("phase-gs.png", GerchbergSaxtonPhase, 255, M, N);
		
		free(GerchbergSaxtonPhase);
		free(target);
	}
	
	return 0;
}

