#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hipfft.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

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
    
typedef double2 complex;

// Allocate double array on device
static double* GPUDouble(int Length, double val, bool init = true)
{
	double* device;

	checkHipErrors(hipMalloc((void**)&device, Length * sizeof(double)));

	if (init)
	{
		auto host = (double*)malloc(Length * sizeof(double));

		if (host != NULL)
		{
			for (int x = 0; x < Length; x++)
			{
				host[x] = val;
			}
		}

		checkHipErrors(hipMemcpy(device, host, Length * sizeof(double), hipMemcpyHostToDevice));

		free(host);
	}

	return device;
}

// Allocate complex array on device
static hipDoubleComplex* GPUComplex(int Length, double val, bool init = true)
{
	hipDoubleComplex* device;

	checkHipErrors(hipMalloc((void**)&device, Length * sizeof(hipDoubleComplex)));

	if (init)
	{
		auto host = (complex*)malloc(Length * sizeof(complex));

		if (host != NULL)
		{
			for (int x = 0; x < Length; x++)
			{
				host[x].x = val;
				host[x].y = 0.0;
			}
		}

		checkHipErrors(hipMemcpy(device, host, Length * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));

		free(host);
	}

	return device;
}

__global__ void KMakeComplexField(double* amplitude, hipDoubleComplex* phase, hipDoubleComplex* ComplexField, int xdim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = x + y * xdim;

	ComplexField[i].x = amplitude[i] * phase[i].x;
	ComplexField[i].y = amplitude[i] * phase[i].y;
}

__global__ void KComputePhase(hipDoubleComplex* field, hipDoubleComplex* phase, int xdim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = x + y * xdim;

	double angle = atan2(field[i].y, field[i].x);

	phase[i].x = cos(angle);
	phase[i].y = sin(angle);
}

// Get angle then wrap
__global__ void KWrap(double* phase, hipDoubleComplex* field, int xdim, double m)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = x + y * xdim;

	double a = atan2(field[i].y, field[i].x);

	phase[i] = a - m * floor(a / m);
}

// Note: Call with all threads assigned to the 1st quadrant (top-left)
__global__ void KShift(double* field, int sizex, int sizey)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double temp;
	int midx = sizex / 2;
	int midy = sizey / 2;

	if (x >= 0 && x < midx && y >= 0 && y < midy)
	{
		int yo = y * sizex;
		int ii = yo + x;

		int ym = (midy + y) * sizex;
		int ymx = ym + x;

		int xo = (midx + x);
		int ymo = ym + xo;
		int yoo = yo + xo;


		// Exchange 1st and 4th quadrant
		temp = field[ii];
		field[ii] = field[ymo];
		field[ymo] = temp;

		// Exchange 2nd and 3rd quadrant
		temp = field[yoo];
		field[yoo] = field[ymx];
		field[ymx] = temp;
	}
}

__global__ void KCreateTargets(double* g, double* phi, hipDoubleComplex* phase, double h, bool gaussian, double r, int aperture, int aperturew, int apertureh, int M, int N)
{
	int xx = blockIdx.x * blockDim.x + threadIdx.x;
	int yy = blockIdx.y * blockDim.y + threadIdx.y;
	int ii = xx + yy * M;

	auto M2 = M / 2;
	auto N2 = N / 2;

	auto aw = aperturew / 2;
	auto ah = apertureh / 2;

	auto eaw = (aperturew * aperturew);
	auto eah = (apertureh * apertureh);

	auto x = xx - M2 + 1;
	auto y = yy - N2 + 1;

	auto GX = h * (double)x / r;
	auto GY = h * (double)y / r;

	double ap = 1.0;

	switch (aperture)
	{
	case 1: // rectangle

		ap = (x * x <= aw * aw && y * y <= ah * ah) ? 1.0 : 0.0;

		break;

	case 2: // ellipse

		ap = (x * x * eah + eaw * y *  y <= eaw * eah) ? 1.0 : 0.0;

		break;

	default: // none

		ap = 1.0;

		break;
	}

	g[ii] = gaussian ? ap * exp(-(GX*GX + GY*GY)) : ap;

	double angle = 2.0 * M_PI * phi[ii];

	phase[ii].x = cos(angle);
	phase[ii].y = sin(angle);
}

bool GPU::IsEnabled()
{
	int nDevices;

	hipGetDeviceCount(&nDevices);

	return nDevices > 0;
}

void GPU::Calculate(double*& GerchbergSaxtonPhase, int M, int N, int Ngs, double h, bool gaussian, double r, int aperture, int aperturew, int apertureh, double*& target)
{
	auto size = M * N;

	// source and target constraints
	auto phaseD = GPUDouble(size, 0.0, false);
	auto g = GPUDouble(size, 0.0, false);
	auto phi = GPUDouble(size, 0.0, false);
	auto phase = GPUComplex(size, 0.0, false);

	// generate random phase
	hiprandGenerator_t gen;
	hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_MT19937);
	hiprandGenerateUniformDouble(gen, phi, size);

	auto const blocksizex = 16;
	auto const blocksizey = 16;

	dim3 dimBlock(blocksizex, blocksizey);
	dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
	dim3 dimGrid2((M / 2) / dimBlock.x, (N / 2) / dimBlock.y);

	// generate source, random phase
	void* KCreateTargetsArgs[] = { &g, &phi, &phase, &h, &gaussian, &r, &aperture, &aperturew, &apertureh, &M, &N };
	checkHipErrors(hipLaunchKernel((const void*)&KCreateTargets, dimGrid, dimBlock, KCreateTargetsArgs));

	// copy target
	auto targetD = GPUDouble(size, 0.0, false);
	checkHipErrors(hipMemcpy(targetD, target, size * sizeof(double), hipMemcpyHostToDevice));

	hipfftHandle plan;

	auto temp = GPUComplex(size, 0.0, false);
	auto forward = GPUComplex(size, 0.0, false);
	auto backward = GPUComplex(size, 0.0, false);
	auto phasef = GPUComplex(size, 0.0, false);

	// 2D FFT plan
	hipfftPlan2d(&plan, M, N, HIPFFT_Z2Z);

	// secret sauce
	void* KShiftArgs[] = { &targetD, &M, &N };
	checkHipErrors(hipLaunchKernel((const void*)&KShift, dimGrid2, dimBlock, KShiftArgs));

	for (int i = 0; i < Ngs; i++)
	{
		// Apply source constraints
		void* SourceArgs[] = { &g, &phase, &temp, &M };
		checkHipErrors(hipLaunchKernel((const void*)&KMakeComplexField, dimGrid, dimBlock, SourceArgs));

		// perform forward transform
		hipfftExecZ2Z(plan, temp, forward, HIPFFT_FORWARD);

		void* ForwardArgs[] = { &forward, &phasef, &M };
		checkHipErrors(hipLaunchKernel((const void*)&KComputePhase, dimGrid, dimBlock, ForwardArgs));

		// Apply target constraints
		void* TargetArgs[] = { &targetD, &phasef, &temp, &M };
		checkHipErrors(hipLaunchKernel((const void*)&KMakeComplexField, dimGrid, dimBlock, TargetArgs));

		// perform backward transform
		hipfftExecZ2Z(plan, temp, backward, HIPFFT_BACKWARD);

		// retrieve phase
		void* BackwardArgs[] = { &backward, &phase, &M };
		checkHipErrors(hipLaunchKernel((const void*)&KComputePhase, dimGrid, dimBlock, BackwardArgs));
	}

	// Compute phase then wrap into 2 * pi
	auto PI2 = 2.0 * M_PI;
	void* KWrapArgs[] = { &phaseD, &phase, &M, &PI2 };
	checkHipErrors(hipLaunchKernel((const void*)&KWrap, dimGrid, dimBlock, KWrapArgs));
	checkHipErrors(hipMemcpy(GerchbergSaxtonPhase, phaseD, size * sizeof(double), hipMemcpyDeviceToHost));

	// Destroy plans, generators
	hipfftDestroy(plan);
	hiprandDestroyGenerator(gen);

	// Clean-Up arrays allocated in GPU
	checkHipErrors(hipFree(phasef));
	checkHipErrors(hipFree(backward));
	checkHipErrors(hipFree(forward));
	checkHipErrors(hipFree(temp));
	checkHipErrors(hipFree(phase));
	checkHipErrors(hipFree(phi));
	checkHipErrors(hipFree(targetD));
	checkHipErrors(hipFree(g));
	checkHipErrors(hipFree(phaseD));
}
