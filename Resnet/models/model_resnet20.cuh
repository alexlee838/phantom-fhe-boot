#pragma once
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <limits>
#include <utility>  
#include "phantom.h"
#include "context.cuh"
#include "dnn.cuh"
#include "bootstrap.cuh"
#include "error_handle.cuh"
#include "timer.h"
#include "weight_loader.cuh"


using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;



// Assuming conv1_w is std::vector<std::vector<std::vector<std::vector<double>>>>

inline void PrintConvWeights(const std::vector<std::vector<std::vector<std::vector<double>>>>& weights, int num_samples = 2) {
	std::cout << "--- Convolution Weight Sample ---\n";
	for (int o = 0; o < std::min(num_samples, (int)weights.size()); ++o) {
		for (int i = 0; i < std::min(num_samples, (int)weights[o].size()); ++i) {
			for (int h = 0; h < std::min(num_samples, (int)weights[o][i].size()); ++h) {
				for (int w = 0; w < std::min(num_samples, (int)weights[o][i][h].size()); ++w) {
					std::cout << "weight[" << o << "][" << i << "][" << h << "][" << w << "] = "
						<< std::setprecision(6) << weights[o][i][h][w] << "\n";
				}
			}
		}
	}
}

inline void PrintBatchNormParams(const BatchNormParams& bn, int num_samples = 5) {
	std::cout << "--- BatchNorm Parameters Sample ---\n";
	for (int i = 0; i < std::min(num_samples, (int)bn.weight.size()); ++i) {
		std::cout << "weight[" << i << "] = " << bn.weight[i]
			<< ", bias[" << i << "] = " << bn.bias[i]
			<< ", mean[" << i << "] = " << bn.mean[i]
			<< ", var[" << i << "] = " << bn.var[i] << "\n";
	}
}



inline std::pair<double, double> find_min_max(const std::vector<std::vector<std::vector<double>>>& tensor3d) {
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();

    for (const auto& matrix : tensor3d) {
        for (const auto& row : matrix) {
            for (double v : row) {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }
    }

    return { min_val, max_val };
}


inline std::pair<double, double> find_abs_min_max(const std::vector<std::vector<std::vector<double>>>& tensor3d) {
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = 0.0;

    for (const auto& matrix : tensor3d) {
        for (const auto& row : matrix) {
            for (double v : row) {
                double abs_v = std::abs(v);
                if (abs_v < min_val) min_val = abs_v;
                if (abs_v > max_val) max_val = abs_v;
            }
        }
    }

    return { min_val, max_val };
}

template<typename T>
inline void print_vector1(std::vector<T> vec, std::size_t print_size = 4, int prec = 3) {
	/*
	Save the formatting information for std::cout.
	*/
	std::ios old_fmt(nullptr);
	old_fmt.copyfmt(std::cout);

	std::size_t slot_count = vec.size();

	std::cout << std::fixed << std::setprecision(prec);
	std::cout << std::endl;
	if (slot_count <= 2 * print_size) {
		std::cout << "    [";
		for (std::size_t i = 0; i < slot_count; i++) {
			std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
		}
	}
	else {
		vec.resize(std::max(vec.size(), 2 * print_size));
		std::cout << "    [";
		for (std::size_t i = 0; i < print_size; i++) {
			std::cout << " " << vec[i] << ",";
		}
		if (vec.size() > 2 * print_size) {
			std::cout << " ...,";
		}
		for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
			std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
		}
	}
	std::cout << std::endl;

	/*
	Restore the old std::cout formatting.
	*/
	std::cout.copyfmt(old_fmt);
}


TensorCT ResNet20_infer(const TensorCT& input, DNN& model, FHECKKSRNS& bootstrapper, const PhantomContext& context, const std::string& weight_dir);
void PrePareResNet20(DNN& model, PhantomSecretKey& secret_key);
void debug_print(const TensorCT& x, DNN* model, PhantomSecretKey* secret_key, PhantomContext* context, PhantomCKKSEncoder* encoder, bool set);