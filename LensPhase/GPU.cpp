#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>

#include "GPU.h"

#define checkHipErrors(cmd)                                                                                 \
{                                                                                              \
	hipError_t error = cmd;                                                                    \
	if (error != hipSuccess) {                                                                 \
		fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
				__FILE__, __LINE__);                                                           \
		exit(EXIT_FAILURE);                                                                    \
	}                                                                                          \
}

// Allocate double array on device
double* GPUDouble(int Length, double val, bool init = true)
{
	double* device;

	checkHipErrors(hipMalloc(&device, Length * sizeof(double)));

	if (init)
	{
		auto host = (double*)malloc(Length * sizeof(double));

		if (host != NULL)
		{
			for (auto x = 0; x < Length; x++)
			{
				host[x] = val;
			}
		}

		checkHipErrors(hipMemcpy(device, host, Length * sizeof(double), hipMemcpyHostToDevice));

		free(host);
	}

	return device;
}

__global__ void KComputePhase(double* phase, double z, double lambda, double h, int M, int N)
{
	int xx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int yy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int ii = xx + yy * M;

	double m = 2.0 * M_PI;
	auto M2 = M / 2;
	auto N2 = N / 2;

	double Nlambda = lambda / (h * h);
	double K0 = m / lambda;
	double KK0 = K0 * K0;

	double x = (double)(xx - M2 + 1);
	double y = (double)(yy - N2 + 1);

	double kkx = pow(x * Nlambda, 2.0);
	double kky = pow(y * Nlambda, 2.0);

	double karg = kkx + kky;

	if (karg <= KK0)
	{
		phase[ii] = sqrt(KK0 - karg) * z;
	}
	else
	{
		phase[ii] = 0.0;
	}
}

bool GPU::IsEnabled()
{
	int nDevices;

	hipGetDeviceCount(&nDevices);

	return nDevices > 0;
}

void GPU::Calculate(double* LensPhase, int M, int N, double z, double lambda, double h)
{
	auto size = M * N;

	// source and target constraints
	auto phaseD = GPUDouble(size, 0.0, false);

	auto const blocksizex = 16;
	auto const blocksizey = 16;

	dim3 dimBlock(blocksizex, blocksizey);
	dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);

	// compute lens phase on the GPU and copy to host
	hipLaunchKernelGGL(KComputePhase, dimGrid, dimBlock, 0, 0, phaseD, z, lambda, h, M, N);
	hipDeviceSynchronize();
	
	checkHipErrors(hipMemcpy(LensPhase, phaseD, size * sizeof(double), hipMemcpyDeviceToHost));

	checkHipErrors(hipFree(phaseD));
}
