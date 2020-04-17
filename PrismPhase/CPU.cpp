#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "CPU.h"

void ComputePhase(double* phase, double MirrorX, double MirrorY, int M, int N)
{
	double m = 2.0 * M_PI;

	for (int yy = 0; yy < N; yy++)
	{
		for (int xx = 0; xx < M; xx++)
		{
			int ii = xx + yy * M;

			double x = xx * MirrorX / 500.0;
			double y = yy * MirrorY / 500.0;

			double angle = x + y;

			phase[ii] = angle - m * floor(angle / m);
		}
	}
}

void CPU::Calculate(double* PrismPhase, int M, int N, double MirrorX, double MirrorY)
{
	ComputePhase(PrismPhase, MirrorX, MirrorY, M, N);
}
