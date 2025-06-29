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
#ifdef _OPENMP
#include <omp.h>
#endif

namespace phantom
{	
	
	class PhantomConvolution
	{
	public:
		PhantomConvolution() = default;
		~PhantomConvolution() = default;

		void zero_pad_encode(const PhantomContext &context, const int f_w, PhantomPublicKey &pk,
							 const std::vector<std::vector<double>> &din, std::vector<PhantomCiphertext> &dout);

		void EvalConvolution(const int in_h, std::vector<std::vector<std::vector<double>>> &filter, const PhantomContext &context,
							 const PhantomGaloisKey &gal_keys, const std::vector<PhantomCiphertext> &din, std::vector<PhantomCiphertext> &dout);

		void SetRotationKeys(const PhantomContext &context, PhantomSecretKey &secret_key, const int in_h, const int f_h);

		void ConvolutionOP(const int in_h, const PhantomContext &context, std::vector<std::vector<std::vector<std::vector<double>>>> &enc_filter,
						   const PhantomCiphertext &din, const std::vector<PhantomPlaintext> &hadamard_ntt, std::vector<std::vector<PhantomCiphertext>> &dout);

		void ConvDecode(const PhantomContext &context, PhantomSecretKey &secret_key, const std::vector<PhantomCiphertext> &dout, std::vector<std::vector<double>> &img_out);

		std::vector<PhantomPlaintext> FCWeightEncodeCore(const PhantomContext &context, const std::vector<std::vector<double>> &weight);

		PhantomPlaintext FCBiasEncodeCore(const PhantomContext &context, const std::vector<double> &bias);

		PhantomCiphertext FullyConnectedLayerCore(const PhantomContext &context, const PhantomGaloisKey &gal_keys, const PhantomCiphertext &din,
												  const std::vector<PhantomPlaintext> &weight, const PhantomPlaintext &bias, const size_t col, bool add_bias = true);

												  void processFullyConnectedLayer(
													const PhantomContext& context,
													const PhantomGaloisKey& galois_keys,
													const std::vector<std::vector<double>>& weight,
													const std::vector<double>& bias,
													const std::vector<PhantomCiphertext>& ct,
													std::vector<PhantomCiphertext>& ct_out,
													int num_of_cipher_in,
													int n_o,
													int n_i);

		inline void setScale(const double scale)
		{
			scale_ = scale;
		}

		// For Debugging
		inline void setSecretKey(PhantomSecretKey &sk)
		{
			secret_keys_ = sk;
		}

		inline size_t smallestPowerOf2LargerThan(size_t n)
		{
			// Special case: if n is 0, the smallest power of 2 larger than 0 is 1.
			// (If you instead want the next power of 2 for n=0 to be 2, adjust as needed.)
			if (n == 0)
			{
				return 1;
			}

			size_t power = 1;
			// Keep shifting left (doubling) until the power exceeds n
			while (power <= n)
			{
				// Watch out for potential overflow if n is very large,
				// but for most practical 32-bit use-cases, this is fine.
				power <<= 1;
			}
			return power;
		}

		// For Debugging
		inline void printCipher(const PhantomCiphertext &din, const PhantomContext &context, const int wPp = 32)
		{
			PhantomCKKSEncoder encoder(context);
			std::vector<double> temp_result;
			PhantomPlaintext temp_plain;
			std::cout << "----------------------------------------------------" << std::endl;
			int cnt = 1;
			secret_keys_.decrypt(context, din, temp_plain);
			encoder.decode(context, temp_plain, temp_result);
			for (auto &val : temp_result)
			{
				std::cout << val << " ";
				if (cnt % wPp == 0)
					std::cout << '\n';
				if (cnt % (wPp * wPp) == 0)
					std::cout << "\n\n\n";
				cnt++;
			}
		}

	private:
		
		PhantomGaloisKeyFused galois_keys_;
		PhantomSecretKey secret_keys_; // Debugging.
		double scale_;
		std::vector<double> m_scalingFactorsReal_;
		std::vector<double> m_scalingFactorsRealBig_;
	};

}
