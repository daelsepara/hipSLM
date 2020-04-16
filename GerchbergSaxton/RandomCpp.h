#pragma once

#include <random>
#include <chrono>

class RandomCpp
{
public:

	std::mt19937_64 generator;
	std::normal_distribution<double> normalDistribution;
	std::uniform_real_distribution<double> uniformDistribution;

	RandomCpp()
	{
		generator = std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count());
	}

	RandomCpp(int seed)
	{
		generator = std::mt19937_64(seed);
	}

	void UniformDistribution(double a, double b)
	{
		uniformDistribution = std::uniform_real_distribution<double>(a, b);
	}

	void UniformDistribution()
	{
		UniformDistribution(0.0, 1.0);
	}

	double NextDouble()
	{
		return uniformDistribution(generator);
	}

	void NormalDistribution(double mean, double std)
	{
		normalDistribution = std::normal_distribution<double>(mean, std);
	}

	void NormalDistribution()
	{
		NormalDistribution(0.0, 1.0);
	}

	double NextNormal()
	{
		return normalDistribution(generator);
	}
};

