#include "dnn_example.h"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#include <iostream>
#include <string>
#include <cstdlib> // For atoi

void ConvolutionExample(int input_width, int filter_width, int input_channels, int output_channels);
void SoftMaxExample(int input_width, int input_channels);

int main(int argc, char **argv)
{
	std::cout << "[Running] ConvolutionExample(inputWidth=32, filterSize=3, inChannels=3, outChannels=16)\n";
	ConvolutionExample(32, 3, 3, 16);

	std::cout << "[Running] ConvolutionExample(inputWidth=32, filterSize=3, inChannels=16, outChannels=16)\n";
	ConvolutionExample(32, 3, 16, 16);

	std::cout << "[Running] ConvolutionExample(inputWidth=16, filterSize=3, inChannels=32, outChannels=32)\n";
	ConvolutionExample(16, 3, 32, 32);

	std::cout << "[Running] ConvolutionExample(inputWidth=16, filterSize=3, inChannels=32, outChannels=64)\n";
	ConvolutionExample(16, 3, 32, 64);

	std::cout << "[Running] ConvolutionExample(inputWidth=8, filterSize=3, inChannels=64, outChannels=64)\n";
	ConvolutionExample(8, 3, 64, 64);

	std::cout << "[Running] ConvolutionExample(inputWidth=16, filterSize=1, inChannels=32, outChannels=64)\n";
	ConvolutionExample(16, 1, 32, 64);

	// SoftMaxExample(1, 10);
	return 0;
}

void SoftMaxExample(int input_width, int input_channels)
{

	int device_id = 0; // Change to the desired GPU index
	cudaSetDevice(device_id);

	EncryptionParameters parameters(scheme_type::ckks);

	size_t N = std::pow(2, 16);

	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;

	auto special_modulus_size = 10; // static_cast<uint32_t>(std::ceil(((firstMod + dcrtBits * (numLargeDigits - 1))) / AUX_MOD));

	std::vector<int> mod_vec = {};
	uint32_t levelsAvailableAfterBootstrap = 11;
	std::vector<uint32_t> levelBudget = {4, 4};

	uint32_t depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget);

	for (int i = 0; i < depth + 1 + special_modulus_size; i++)
	{
		if (i == 0)
		{
			mod_vec.push_back(firstMod);
		}

		else
		{
			if (i < depth + 1)
			{
				mod_vec.push_back(dcrtBits);
			}
			else
			{
				mod_vec.push_back(AUX_MOD);
			}
		}
	}

	parameters.set_poly_modulus_degree(N);
	parameters.set_special_modulus_size(special_modulus_size);
	parameters.set_coeff_modulus(CoeffModulus::Create(N, mod_vec));
	double scale = pow(2.0, 59);

	Timer::startGPUTimer("Context Creation");
	PhantomContext context(parameters);
	Timer::stopGPUTimer("Context Creation");

	PhantomSecretKey secret_key(context);
	PhantomPublicKey public_key = secret_key.gen_publickey(context);
	PhantomCKKSEncoder encoder(context);

	int w = input_width;
	int c_i = input_channels;

	std::vector<std::vector<std::vector<double>>> image(w, std::vector<std::vector<double>>(w, std::vector<double>(c_i, 0.0)));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-1.0, 1.0); // Values between 0.0 and 1.0

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int k = 0; k < c_i; k++)
			{
				image[i][j][k] = dist(gen); // Assign a random double
			}
		}
	}

	encoder.set_sparse_encode(context.get_context_data(0).parms(), w * w * 2);

	DNN model(scale, encoder, context);

	Timer::startGPUTimer("Matrix Encode");
	TensorCT input = model.EncTensor(image, public_key);
	Timer::stopGPUTimer("Matrix Encode");

	input.ctks_[0].PreComputeScale(context, scale);

	std::vector<double> m_scalingFactorsReal = input.ctks_[0].getScalingFactorsReal();
	std::vector<double> m_scalingFactorsRealBig = input.ctks_[0].getScalingFactorsRealBig();
	model.setScalingFactors(m_scalingFactorsReal, m_scalingFactorsRealBig);
	PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

	FHECKKSRNS bootstrapper(encoder);

	Timer::startGPUTimer("SoftMax");
	TensorCT output = model.SoftMax(input, relin_keys, bootstrapper);
	Timer::stopGPUTimer("SoftMax");

	PhantomPlaintext result_plain;
	Timer::startGPUTimer("Decryption");
	result_plain = secret_key.decrypt(context, output.ctks_[0]);
	Timer::stopGPUTimer("Decryption");

	std::vector<double> result;
	encoder.decode(context, result_plain, result);
	std::cout << "Result vector: " << std::endl;
	print_vector(result, 3, 7);

	Timer::printAccumulatedTimes();
	Timer::clearAllTimings();
}

void ConvolutionExample(int input_width, int filter_width, int input_channels, int output_channels)
{

	int device_id = 0; // Change to the desired GPU index
	cudaSetDevice(device_id);

	EncryptionParameters parameters(scheme_type::ckks);

	size_t N = std::pow(2, 16);

	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;

	auto special_modulus_size = 10; // static_cast<uint32_t>(std::ceil(((firstMod + dcrtBits * (numLargeDigits - 1))) / AUX_MOD));

	std::vector<int> mod_vec = {};
	uint32_t levelsAvailableAfterBootstrap = 11;
	std::vector<uint32_t> levelBudget = {2, 2};

	uint32_t depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget);
	std::cout << "sizeQ : " << depth + 1 << std::endl;
	for (int i = 0; i < depth + 1 + special_modulus_size; i++)
	{
		if (i == 0)
		{
			mod_vec.push_back(firstMod);
		}

		else
		{
			if (i < depth + 1)
			{
				mod_vec.push_back(dcrtBits);
			}
			else
			{
				mod_vec.push_back(AUX_MOD);
			}
		}
	}

	parameters.set_poly_modulus_degree(N);
	parameters.set_special_modulus_size(special_modulus_size);
	parameters.set_coeff_modulus(CoeffModulus::Create(N, mod_vec));
	double scale = pow(2.0, 59);

	Timer::startGPUTimer("Context Creation");
	PhantomContext context(parameters);
	Timer::stopGPUTimer("Context Creation");

	PhantomSecretKey secret_key(context);
	PhantomPublicKey public_key = secret_key.gen_publickey(context);
	PhantomCKKSEncoder encoder(context);

	std::vector<PhantomCiphertext> ct(output_channels);
	std::vector<PhantomCiphertext> ct_out(output_channels);

	int w = input_width;
	int f_w = filter_width;
	int c_i = input_channels;
	int c_o = output_channels;
	std::vector<std::vector<std::vector<double>>> image(w, std::vector<std::vector<double>>(w, std::vector<double>(c_i, 0.0)));

	std::vector<std::vector<std::vector<std::vector<double>>>> filter(
		f_w,
		std::vector<std::vector<std::vector<double>>>(
			f_w,
			std::vector<std::vector<double>>(
				c_i,
				std::vector<double>(c_o, 1.0))));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-1.0, 1.0); // Values between 0.0 and 1.0

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int k = 0; k < c_i; k++)
			{
				image[i][j][k] = dist(gen); // Assign a random double
			}
		}
	}

	for (int i = 0; i < f_w; i++)
	{
		for (int j = 0; j < f_w; j++)
		{
			for (int k = 0; k < c_i; k++)
			{
				for (int l = 0; l < c_o; l++)
				{
					filter[i][j][k][l] = dist(gen); // Assign a random double
				}
			}
		}
	}
	encoder.set_sparse_encode(context.get_context_data(0).parms(), w * w * 2);

	DNN model(scale, encoder, context);
	model.ComputeRotationIndices(secret_key, input_width, f_w, 0);

	Timer::startGPUTimer("Matrix Encode");
	TensorCT input = model.EncTensor(image, public_key);
	Timer::stopGPUTimer("Matrix Encode");

	input.ctks_[0].PreComputeScale(context, scale);

	std::vector<double> m_scalingFactorsReal = input.ctks_[0].getScalingFactorsReal();
	std::vector<double> m_scalingFactorsRealBig = input.ctks_[0].getScalingFactorsRealBig();
	model.setScalingFactors(m_scalingFactorsReal, m_scalingFactorsRealBig);

	Timer::startGPUTimer("Convolution");
	TensorCT output = model.Conv(input, filter, 1);
	Timer::stopGPUTimer("Convolution");
	FHECKKSRNS bootstrapper(encoder);

	Timer::startGPUTimer("Bootstrap Setup");
	bootstrapper.EvalBootstrapSetup(context, levelBudget, scale, m_scalingFactorsReal, m_scalingFactorsRealBig, {0, 0}, w * w);
	Timer::stopGPUTimer("Bootstrap Setup");

	Timer::startGPUTimer("Multiplication KeyGen");
	bootstrapper.EvalMultKeyGen(secret_key, context);
	Timer::stopGPUTimer("Multiplication KeyGen");

	Timer::startGPUTimer("Bootstrap KeyGen");
	bootstrapper.EvalBootstrapKeyGen(secret_key, context, w * w);
	Timer::stopGPUTimer("Bootstrap KeyGen");

	PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

	PhantomPlaintext result_plain;
	std::vector<double> result;

	result_plain = secret_key.decrypt(context, output.ctks_[0]);
	encoder.decode(context, result_plain, result);

	std::cout << "before relu vector: " << std::endl;
	print_vector(result, 4, 7);

	Timer::startGPUTimer("Relu");
    output = model.ReluComposite(output, bootstrapper);

	Timer::stopGPUTimer("Relu");

	result_plain = secret_key.decrypt(context, output.ctks_[0]);
	encoder.decode(context, result_plain, result);

	std::cout << "after relu vector: " << std::endl;
	print_vector(result, 4, 7);

	std::cout << "Before Bootstrapping : " << mod_vec.size() - output.ctks_[0].chain_index() - special_modulus_size - 1 << std::endl;

	PhantomCiphertext result_cipher;

	Timer::startGPUTimer("Bootstrapping");
	for (int i = 0; i < output_channels; i++)
	{
		ct_out[i] = bootstrapper.EvalBootstrap(output.ctks_[i], context, w * w);
	}
	Timer::stopGPUTimer("Bootstrapping");

	std::cout << "After Bootstrapping : " << mod_vec.size() - output.ctks_[0].chain_index() - special_modulus_size - 1 << std::endl;

	// PhantomPlaintext result_plain;
	Timer::startGPUTimer("Decryption");
	result_plain = secret_key.decrypt(context, ct_out[0]);
	Timer::stopGPUTimer("Decryption");

	// std::vector<double> result;
	encoder.decode(context, result_plain, result);
	std::cout << "Result vector: " << std::endl;
	print_vector(result, 4, 7);

	Timer::printAccumulatedTimes();
	Timer::clearAllTimings();
}
