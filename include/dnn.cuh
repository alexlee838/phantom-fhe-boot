#pragma once

#include <unordered_set>
#include <fstream>
#include "ciphertext.h"
#include "context.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "cuda_wrapper.cuh"
#include "ckks.h"
#include "evaluate.cuh"
#include "timer.h"
#include "bootstrap.cuh"

#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call)                                          \
	do                                                            \
	{                                                             \
		cudaError_t err = call;                                   \
		if (err != cudaSuccess)                                   \
		{                                                         \
			fprintf(stderr, "CUDA Error at %s:%d: %s\n",          \
					__FILE__, __LINE__, cudaGetErrorString(err)); \
			std::exit(EXIT_FAILURE);                              \
		}                                                         \
	} while (0)

namespace phantom
{
	struct TensorCT
	{
		std::vector<PhantomCiphertext> ctks_;
		size_t width_;
		size_t slotstr_;
		size_t num_ch_;
	};

	class DNN
	{
	public:
		DNN(size_t scale, PhantomCKKSEncoder& encoder, const PhantomContext& context)
			: scale_(scale), encoder_(encoder), context_(context)
		{
		}
		~DNN() = default;
		TensorCT EncTensor(const std::vector<std::vector<std::vector<double>>>& input, PhantomPublicKey& pk);
		PhantomPlaintext PlainTextEncode(const std::vector<double>& input, size_t chain_index);
		void ComputeRotationIndices(PhantomSecretKey& secret_key, size_t input_width, size_t kernel_h, size_t slotstr);
		inline void setScalingFactors(const std::vector<double>& scalingFactorsReal, const std::vector<double>& scalingFactorsRealBig)
		{
			m_scalingFactorsReal_ = scalingFactorsReal;
			m_scalingFactorsRealBig_ = scalingFactorsRealBig;
		}
		TensorCT Conv(const TensorCT& input, const std::vector<std::vector<std::vector<std::vector<double>>>>& weight, size_t stride);
		TensorCT Relu(const TensorCT& input, const PhantomRelinKey& relin_keys, int a = -1, int b = 1, int deg = 7);
		TensorCT ReluDegTwo(const TensorCT& input, const PhantomRelinKey& relin_keys);
		TensorCT BootStrap(TensorCT& input, FHECKKSRNS& bootstrapper);
		TensorCT SoftMax(const TensorCT& input, const PhantomRelinKey& relin_keys, FHECKKSRNS& bootstrapper, size_t BoundB = 64, size_t BoundR = 10000, size_t GumbelLambda = 4, size_t GoldSchmidtD = 16);
		TensorCT AvgPoolFullCon(const TensorCT& input, const std::vector<std::vector<double>>& weight, const std::vector<double>& bias);
		TensorCT ReluComposite(const TensorCT& input, FHECKKSRNS& bootstrapper);
		TensorCT BatchNorm(const TensorCT& input, const std::vector<double>& weight, const std::vector<double>& bias, const std::vector<double>& mean, const std::vector<double>& var);
		TensorCT Add(const TensorCT& a, const TensorCT& b);
		TensorCT Sign(const TensorCT& input, const PhantomRelinKey& relin_keys, size_t k);
		void RelinKeyGen(PhantomSecretKey& secret_key);
		void BuildGaloisKey(PhantomSecretKey& secret_key, std::vector<int32_t>& rotation_indices);
		void AddRotationIndicesTo(std::vector<int32_t>& rotation_indices, size_t input_width, size_t kernel_h, size_t slotstr);
		void AddAvgPoolRotationsTo(std::vector<int32_t>& rotation_indices, size_t input_width, size_t slotstr);
		std::vector<std::vector<std::vector<double>>> DecTensor(const TensorCT& enc_tensor, PhantomSecretKey& sk);
	private:
		size_t scale_;
		PhantomGaloisKeyFused gk_;
		const PhantomContext& context_;
		PhantomCKKSEncoder& encoder_;
		std::vector<double> m_scalingFactorsReal_;
		std::vector<double> m_scalingFactorsRealBig_;
		PhantomRelinKey mul_key_;

	};
}
