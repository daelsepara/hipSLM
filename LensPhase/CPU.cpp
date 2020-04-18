#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "CPU.h"

double Mod(double a, double m)
{
	return a - m * floor(a / m);
}

void ComputePhase(double* phase, double z, double lambda, double h, int M, int N)
{
	auto Nlambda = lambda / (h * h);
	auto K0 = 2 * M_PI / lambda;
	auto KK0 = K0 * K0;
	auto M_2 = M / 2;
	auto N_2 = N / 2;

	for (auto y = 0; y < N; y++)
	{
		for (auto x = 0; x < M; x++)
		{
			auto kkx = pow((double)(x - M_2) * Nlambda, 2.0);
			auto kky = pow((double)(y - N_2) * Nlambda, 2.0);
			auto karg = kkx + kky;
			
			if (karg <= KK0)
			{
				phase[y * M + x] = Mod(sqrt(KK0 - karg) * z, 2 * M_PI);
			}
			else
			{
				phase[y * M + x] = 0.0;
			}
		}
	}
}

void CPU::Calculate(double* LensPhase, int M, int N, double z, double lambda, double h)
{
	ComputePhase(LensPhase, z, lambda, h, M, N);
}
