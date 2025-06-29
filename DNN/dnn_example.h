#pragma once

#include <iomanip>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include "phantom.h"
#include "context.cuh"
#include "dnn.cuh"
#include "bootstrap.cuh"
#include "error_handle.cuh"
#include "timer.h"

#define AUX_MOD 60


inline void convolution(
	const std::vector<std::vector<std::vector<double>>> &image,				  // [H][W][C_in]
	std::vector<std::vector<double>> &dout,									  // [H_out * W_out][C_out]
	const std::vector<std::vector<std::vector<std::vector<double>>>> &filter, // [F_H][F_W][C_in][C_out]
	int c_i, int c_o, int in_h, int f_h, int stride)
{
	int pad = (f_h - 1) / 2;
	int out_h = (in_h + 2 * pad - f_h) / stride + 1;

	dout.resize(out_h * out_h, std::vector<double>(c_o, 0.0)); // Allocate output buffer

	for (int i = 0; i < out_h; i++)
	{
		for (int j = 0; j < out_h; j++)
		{
			for (int oc = 0; oc < c_o; oc++)
			{
				double sum = 0.0;

				for (int ic = 0; ic < c_i; ic++)
				{
					for (int fi = 0; fi < f_h; fi++)
					{
						for (int fj = 0; fj < f_h; fj++)
						{
							int row = i * stride + fi - pad;
							int col = j * stride + fj - pad;

							double pixel = (row >= 0 && row < in_h && col >= 0 && col < in_h)
											   ? image[row][col][ic]
											   : 0.0;

							sum += pixel * filter[fi][fj][ic][oc];
						}
					}
				}

				dout[i * out_h + j][oc] = sum; // Flattened indexing
			}
		}
	}
}

inline bool areVectorsIdentical(const std::vector<double> &vec1,
								const std::vector<double> &vec2,
								double epsilon = 1e-6)
{
	if (vec1.size() != vec2.size())
		return false;

	for (size_t i = 0; i < vec1.size(); i++)
	{
		if (std::fabs(vec1[i] - vec2[i]) > epsilon)
			return false;
	}
	return true;
}

template <typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4, int prec = 3)
{
	/*
	Save the formatting information for std::cout.
	*/
	std::ios old_fmt(nullptr);
	old_fmt.copyfmt(std::cout);

	std::size_t slot_count = vec.size();

	std::cout << std::fixed << std::setprecision(prec);
	std::cout << std::endl;
	if (slot_count <= 2 * print_size)
	{
		std::cout << "    [";
		for (std::size_t i = 0; i < slot_count; i++)
		{
			std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
		}
	}
	else
	{
		vec.resize(std::max(vec.size(), 2 * print_size));
		std::cout << "    [";
		for (std::size_t i = 0; i < print_size; i++)
		{
			std::cout << " " << vec[i] << ",";
		}
		if (vec.size() > 2 * print_size)
		{
			std::cout << " ...,";
		}
		for (std::size_t i = slot_count - print_size; i < slot_count; i++)
		{
			std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
		}
	}
	std::cout << std::endl;

	/*
	Restore the old std::cout formatting.
	*/
	std::cout.copyfmt(old_fmt);
}
