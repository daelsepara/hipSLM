#pragma once

class GPU
{
public:

	bool IsEnabled();

	void Calculate(double* GerchbergSaxtonPhase, int M, int N, int Ngs, double h, bool gaussian, double r, int aperture, int aperturew, int apertureh, double* target);
};
