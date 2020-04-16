#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>

#include "CPU.h"
#include "RandomCpp.h"

static const int Re = 0;
static const int Im = 1;

static void MakeComplexField(double* amplitude, fftw_complex* phase, fftw_complex* ComplexField, int size)
{
	for (int index = 0; index < size; index++)
	{
		ComplexField[index][Re] = amplitude[index] * phase[index][Re];
		ComplexField[index][Im] = amplitude[index] * phase[index][Im];
	}
}

static void ComputePhase(fftw_complex* field, fftw_complex* phase, int size)
{
	for (int index = 0; index < size; index++)
	{
		auto angle = std::atan2(field[index][Im], field[index][Re]);

		phase[index][Re] = std::cos(angle);
		phase[index][Im] = std::sin(angle);
	}
}

static double Mod(double a, double m)
{
	return a - m * floor(a / m);
}

static double* Double(int Length, double val)
{
	auto dbl = (double*)malloc(Length * sizeof(double));

	if (dbl != NULL)
	{
		for (auto x = 0; x < Length; x++)
		{
			dbl[x] = val;
		}
	}

	return dbl;
}

static fftw_complex* Complex(int Length, double val)
{
	auto dbl = (fftw_complex*)fftw_malloc(Length * sizeof(fftw_complex));

	if (dbl != NULL)
	{
		for (auto x = 0; x < Length; x++)
		{
			dbl[x][Re] = val;
			dbl[x][Im] = 0.0;
		}
	}

	return dbl;
}

static void Shift(double* field, int sizex, int sizey)
{
	double temp;
	auto midx = sizex / 2;
	auto midy = sizey / 2;

	for (auto y = 0; y < midy; y++)
	{
		for (auto x = 0; x < midx; x++)
		{
			// Exchange 1st and 4th quadrant
			temp = field[y * sizex + x];
			field[y * sizex + x] = field[(midy + y) * sizex + (midx + x)];
			field[(midy + y) * sizex + (midx + x)] = temp;

			// Exchange 2nd and 3rd quadrant
			temp = field[y * sizex + (midx + x)];
			field[y * sizex + (midx + x)] = field[(midy + y) * sizex + x];
			field[(midy + y) * sizex + x] = temp;
		}
	}
}

void CPU::Calculate(double* GerchbergSaxtonPhase, int M, int N, int Ngs, double h, bool gaussian, double r, int aperture, int aperturew, int apertureh, double* target)
{
	RandomCpp random = RandomCpp();

	auto size = M * N;

	// source
	auto g = Double(size, 0.0);
	
	// random initial phase
	auto phase = Complex(size, 0.0);

	auto N2 = N / 2;
	auto M2 = M / 2;

	auto aw = aperturew / 2;
	auto ah = apertureh / 2;

	auto eaw = (aperturew * aperturew);
	auto eah = (apertureh * apertureh);

	// generate source, aperture, initial phase
	for (int y = (1 - N2); y < N2 + 1; y++)
	{
		for (int x = (1 - M2); x < M2 + 1; x++)
		{
			auto xx = x + M2 - 1;
			auto yy = y + N2 - 1;

			auto ii = yy * M + xx;

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

			g[ii] = gaussian ? ap * std::exp(-(GX*GX + GY*GY)) : ap;
			
			auto phi = random.NextDouble() * 2.0 * M_PI;
			
			phase[ii][Re] = std::cos(phi);
			phase[ii][Im] = std::sin(phi);
		}
	}

	auto temp = Complex(size, 0.0);
	auto forward = Complex(size, 0.0);
	auto backward = Complex(size, 0.0);
	auto phasef = Complex(size, 0.0);

	// FFT plans
	fftw_plan fwdplan, invplan;

	// 2D Forward plan
	fwdplan = fftw_plan_dft_2d(N, M, forward, backward, FFTW_FORWARD, FFTW_ESTIMATE);

	// 2D Inverse plan
	invplan = fftw_plan_dft_2d(N, M, backward, forward, FFTW_BACKWARD, FFTW_ESTIMATE);

	// secret sauce
	Shift(target, M, N);

	for (int i = 0; i < Ngs; i++)
	{
		// Apply source constraints
		MakeComplexField(g, phase, temp, size);

		// perform forward transform
		fftw_execute_dft(fwdplan, temp, forward);

		ComputePhase(forward, phasef, size);

		// Apply target constraints
		MakeComplexField(target, phasef, temp, size);

		// perform backward transform
		fftw_execute_dft(invplan, temp, backward);

		// retrieve phase
		ComputePhase(backward, phase, size);
	}

	auto PI2 = 2.0 * M_PI;

	for (int index = 0; index < size; index++)
	{
		GerchbergSaxtonPhase[index] = Mod(atan2(phase[index][Im], phase[index][Re]), PI2);
	}

	// Destroy plans
	fftw_destroy_plan(invplan);
	fftw_destroy_plan(fwdplan);

	// FFTW clean-up
	fftw_cleanup();

	fftw_free(phasef);
	fftw_free(backward);
	fftw_free(forward);
	fftw_free(temp);
	fftw_free(phase);
	free(g);
}
