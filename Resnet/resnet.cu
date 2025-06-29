#include "resnet.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#include <iostream>
#include <string>
#include <cstdlib> // For atoi

int main(int argc, char** argv)
{
	int device_id = 1; // Change to the desired GPU index
	cudaSetDevice(device_id);

	EncryptionParameters parameters(scheme_type::ckks);

	size_t N = std::pow(2, 16);

	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;

	auto special_modulus_size = 10; // static_cast<uint32_t>(std::ceil(((firstMod + dcrtBits * (numLargeDigits - 1))) / AUX_MOD));

	std::vector<int> mod_vec = {};
	uint32_t levelsAvailableAfterBootstrap = 11;
	std::vector<uint32_t> levelBudget = { 2, 2 };

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

	int w = 32;
	int f_w = 3;
	int c_i = 3;

	std::vector<std::vector<std::vector<double>>> image(w, std::vector<std::vector<double>>(w, std::vector<double>(c_i, 0.0)));


	image = load_next_cifar_image("/home/student/temp/shlee/CKKS_Bootstrapping/Resnet/input_image/cifar10_test_images_float32.npy");
	encoder.set_sparse_encode(context.get_context_data(0).parms(), w * w * 2);





	DNN model(scale, encoder, context);
	model.ComputeRotationIndices(secret_key, w, f_w, 0);

	TensorCT input = model.EncTensor(image, public_key);

	input.ctks_[0].PreComputeScale(context, scale);

	std::vector<double> m_scalingFactorsReal = input.ctks_[0].getScalingFactorsReal();
	std::vector<double> m_scalingFactorsRealBig = input.ctks_[0].getScalingFactorsRealBig();
	model.setScalingFactors(m_scalingFactorsReal, m_scalingFactorsRealBig);

	PrePareResNet20(model, secret_key);


	FHECKKSRNS bootstrapper(encoder);
	bootstrapper.EvalBootstrapSetup(context, levelBudget, scale, m_scalingFactorsReal, m_scalingFactorsRealBig, { 0, 0 }, w * w);
	bootstrapper.EvalMultKeyGen(secret_key, context);
	bootstrapper.EvalBootstrapKeyGen(secret_key, context, w * w);
	debug_print(input, &model, &secret_key, &context, &encoder, true);
	
	std::cout << "Inference Start" << std::endl;
	Timer::startGPUTimer("Resnet20 Inference");
	TensorCT output = ResNet20_infer(input, model, bootstrapper, context, "/home/student/temp/shlee/CKKS_Bootstrapping/Resnet/resnet20_weights");
	Timer::stopGPUTimer("Resnet20 Inference");
	std::cout << "Inference End" << std::endl;	

	auto result = model.DecTensor(output, secret_key);
	std::cout << "Result vector: " << std::endl;
	auto [min_val, max_val] = find_min_max(result);

	std::cout << "Min: " << min_val << ", Max: " << max_val << std::endl;

	Timer::printAccumulatedTimes();
	Timer::clearAllTimings();

	return 0;
}