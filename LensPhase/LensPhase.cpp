// LensPhase.cpp : Defines the exported functions for the DLL application.
//
#include <cstdio>
#include <cstdlib>

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
		if (argc >= 7)
		{
			auto LensPhase = (double*)(argv[0]);  // destination
			auto M = *((int*)(argv[1]));  // SLM width in # of pixels
			auto N = *((int*)(argv[2]));  // SLM height in # of pixels
			auto z = *((double*)(argv[3])) * 1e-6; // z
			auto lambda = *((double*)(argv[4])) * 1e-9;  // lambda
			auto h = *((double*)(argv[5])) * 1e-6; // SLM pixel size
			auto useGPU = *((bool*)(argv[6]));  // Force GPU

			auto SLM = M >= N ? M : N;

			auto temp = Double(SLM * SLM, 0.0);

			CPU cpu;
			GPU gpu;

			if (useGPU && gpu.IsEnabled())
			{
				gpu.Calculate(temp, SLM, SLM, z, lambda, h);
			}
			else
			{
				cpu.Calculate(temp, SLM, SLM, z, lambda, h);
			}

			auto cx = (SLM - M) / 2;
			auto cy = (SLM - N) / 2;

			for (int y = 0; y < N; y++)
			{
				for (int x = 0; x < M; x++)
				{
					auto dst = x + y * M;
					auto src = (cx + x) + (cy + y) * SLM;

					LensPhase[dst] = temp[src];
				}
			}

			free(temp);
		}
	}
}
