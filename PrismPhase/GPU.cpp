#define _USE_MATH_DEFINES
#include <math.h>
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

__global__ void KComputePhase(double* phase, double MirrorX, double MirrorY, int M, int N)
{
	int xx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int yy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int ii = xx + yy * M;

	double m = 2.0 * M_PI;
	double x = xx * MirrorX / 500.0;
	double y = yy * MirrorY / 500.0;

	double angle = x + y;

	phase[ii] = angle - m * floor(angle / m);
}

bool GPU::IsEnabled()
{
	int nDevices;

	hipGetDeviceCount(&nDevices);

	return nDevices > 0;
}

void GPU::Calculate(double* PrismPhase, int M, int N, double MirrorX, double MirrorY)
{
	auto size = M * N;

	// source and target constraints
	auto phaseD = GPUDouble(size, 0.0, false);

	auto const blocksizex = 16;
	auto const blocksizey = 16;

	dim3 dimBlock(blocksizex, blocksizey);
	dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);

	// compute blazed grating phase on the GPU and copy to host
	hipLaunchKernelGGL(KComputePhase, dimGrid, dimBlock, 0, 0, phaseD, MirrorX, MirrorY, M, N);
	hipDeviceSynchronize();
	
	checkHipErrors(hipMemcpy(PrismPhase, phaseD, size * sizeof(double), hipMemcpyDeviceToHost));

	checkHipErrors(hipFree(phaseD));
}
