// GerchbergSaxton.cpp : Defines the exported functions for the DLL application.
//
#define _USE_MATH_DEFINES
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>

#include "CPU.h"
#include "GPU.h"

#define DLL_PUBLIC __attribute__((visibility("default")))

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

extern "C"
{
	DLL_PUBLIC void Calculate(int argc, void** argv)
	{
		if (argc >= 14)
		{
			auto GerchbergSaxtonPhase = (double*)(argv[0]);  // destination
			auto M = *((int*)(argv[1]));  // SLM width in # of pixels
			auto N = *((int*)(argv[2]));  // SLM height in # of pixels
			auto Ngs = *((int*)(argv[3]));  // Ngs
			auto h = *((double*)(argv[4]));  // hologram pixel size
			auto gaussian = *((bool*)(argv[5]));  // Gaussian Illumination
			auto r = *((double*)(argv[6]));  // input Gaussian beam waist
			auto aperture = *((int*)(argv[7]));  // aperture type (0 - None, 1 - rectangle, 2 - ellipse)
			auto aperturew = *((int*)(argv[8]));  // aperture Width
			auto apertureh = *((int*)(argv[9]));  // aperture Height
			auto target = (double*)(argv[10]);  // target
			auto targetw = *((int*)(argv[11]));  // target width
			auto targeth = *((int*)(argv[12]));  // target height
			auto useGPU = *((bool*)(argv[13]));  // Force GPU

			// determine the optimal size of computation to use
			auto TSize = targetw >= targeth ? targetw : targeth;
			auto SDim = M >= N ? M : N;
			auto SLM = SDim >= TSize ? SDim : TSize;

			// use a zero-padded target
			auto TPad = Double(SLM * SLM, 0.0);

			auto tx = (SLM - targetw) / 2;
			auto ty = (SLM - targeth) / 2;

			for (int y = 0; y < targeth; y++)
			{
				for (int x = 0; x < targetw; x++)
				{
					auto dst = (tx + x) + (ty + y) * SLM;
					auto src = x + y * targetw;

					TPad[dst] = target[src];
				}
			}

			auto temp = Double(SLM * SLM, 0.0);

			CPU cpu;
			GPU gpu;

			// Compute phase using the Gerchberg-Saxton algorithm
			if (useGPU && gpu.IsEnabled())
			{
				gpu.Calculate(temp, SLM, SLM, Ngs, h, gaussian, r, aperture, aperturew, apertureh, TPad);
			}
			else
			{
				cpu.Calculate(temp, SLM, SLM, Ngs, h, gaussian, r, aperture, aperturew, apertureh, TPad);
			}

			auto cx = (SLM - M) / 2;
			auto cy = (SLM - N) / 2;

			for (int y = 0; y < N; y++)
			{
				for (int x = 0; x < M; x++)
				{
					auto dst = x + y * M;
					auto src = (cx + x) + (cy + y) * SLM;

					GerchbergSaxtonPhase[dst] = temp[src];
				}
			}

			free(TPad);
			free(temp);
		}
	}
}
