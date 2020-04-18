#pragma once

class GPU
{
public:

	bool IsEnabled();

	void Calculate(double* LensPhase, int M, int N, double z, double lambda, double h);
};
