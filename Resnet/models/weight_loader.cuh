#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

#include <string>
#include <vector>
#include "cnpy.h"  // Required for npy loading

using Weight4D = std::vector<std::vector<std::vector<std::vector<double>>>>;

// Struct for BatchNorm parameters
struct BatchNormParams {
	std::vector<double> weight;
	std::vector<double> bias;
	std::vector<double> mean;
	std::vector<double> var;
};

// Loaders
Weight4D LoadWeight4D(const std::string& path);
std::vector<std::vector<double>> LoadWeight2D(const std::string& path);
std::vector<double> LoadWeight1D(const std::string& path);
BatchNormParams LoadBN(const std::string& dir, const std::string& prefix);
std::vector<std::vector<std::vector<double>>> load_next_cifar_image(const std::string& npy_path);
#endif // WEIGHT_LOADER_HPP
