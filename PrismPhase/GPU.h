#pragma once

class GPU
{
public:

	bool IsEnabled();

	void Calculate(double* PrismPhase, int M, int N, double MirrorX, double MirrorY);
};
