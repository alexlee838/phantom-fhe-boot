#include "dnn.cuh"
using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom
{

	TensorCT DNN::EncTensor(const std::vector<std::vector<std::vector<double>>>& input, PhantomPublicKey& pk)
	{
		TensorCT result;
		result.width_ = input.size();
		result.num_ch_ = input[0][0].size();

		auto& parms = context_.get_context_data(context_.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();
		result.slotstr_ = 0;

		for (size_t k = 0; k < result.num_ch_; k++)
		{
			std::vector<double> vk(result.width_ * result.width_, 0.0);
			for (size_t i = 0; i < result.width_; i++)
			{
				for (size_t j = 0; j < result.width_; j++)
				{
					size_t x = i * result.width_ + j;
					vk[x] = input[i][j][k];
				}
			}

			PhantomPlaintext x_plain;
			PhantomCiphertext x_cipher;
			encoder_.encode_sparse(context_, vk, scale_, x_plain, 1);
			pk.encrypt_asymmetric(context_, x_plain, x_cipher);
			result.ctks_.push_back(x_cipher);
		}

		return result;
	}

	std::vector<std::vector<std::vector<double>>> DNN::DecTensor(const TensorCT& enc_tensor, PhantomSecretKey& sk)
	{
		size_t width = enc_tensor.width_;
		size_t num_ch = enc_tensor.num_ch_;
		std::vector<std::vector<std::vector<double>>> output(width, std::vector<std::vector<double>>(width, std::vector<double>(num_ch)));

		for (size_t k = 0; k < num_ch; k++)
		{
			PhantomPlaintext x_plain;
			std::vector<double> vk;

			// Decrypt
			x_plain = sk.decrypt(context_, enc_tensor.ctks_[k]);

			// Decode
			encoder_.decode_sparse(context_, x_plain, (width << enc_tensor.slotstr_) * (width << enc_tensor.slotstr_), vk);

			// Reshape back to [width][width] for channel k
			for (size_t i = 0; i < width; i++)
			{
				for (size_t j = 0; j < width; j++)
				{
					size_t x = i * width + j;
					output[i][j][k] = vk[x];
				}
			}
		}

		return output;
	}



	PhantomPlaintext DNN::PlainTextEncode(const std::vector<double>& input, size_t chain_index)
	{
		PhantomPlaintext x_plain;
		encoder_.encode_sparse(context_, input, scale_, x_plain, chain_index);
		return x_plain;
	}

	TensorCT DNN::Conv(const TensorCT& input, const std::vector<std::vector<std::vector<std::vector<double>>>>& weight, size_t stride)
	{
		Timer::startGPUTimer("Convolution");
		int out_ch = weight[0][0][0].size();
		int in_ch = weight[0][0].size();
		int kernel_h = weight[0].size();

		TensorCT result;
		assert(stride == 1 || stride == 2);

		int pow_slotstr = std::pow(2, input.slotstr_);
		int large_L = input.width_ * pow_slotstr;
		for (int h = 0; h < out_ch; h++)
		{
			bool first_flag = true;
			for (int k = 0; k < in_ch; k++)
			{
				for (int j = 0; j < kernel_h; j++)
				{
					for (int i = 0; i < kernel_h; i++)
					{
						std::vector<double> weight_sp(large_L * large_L, 0.0f);
						for (int j_prime = 0; j_prime < input.width_; j_prime++)
						{
							for (int i_prime = 0; i_prime < input.width_; i_prime++)
							{
								if ((i_prime + i - kernel_h / 2 <= input.width_ - 1) && (j_prime + j - kernel_h / 2 <= input.width_ - 1) && (i_prime + i - kernel_h / 2 >= 0) && (j_prime + j - kernel_h / 2 >= 0))
								{
									weight_sp[(i_prime * large_L + j_prime) * pow_slotstr] = weight[i][j][k][h];
								}
							}
						}
						int rot = (i - kernel_h / 2) * large_L + j - kernel_h / 2;
						if (first_flag)
						{
							PhantomCiphertext input_cipher_k = input.ctks_[k];

							PhantomCiphertext input_cipher_k_rot;

							PhantomPlaintext weight_plain = PlainTextEncode(weight_sp, input_cipher_k.chain_index());
							EvalRotateFused(context_, gk_, input_cipher_k, input_cipher_k_rot, rot * pow_slotstr);
							EvalMultAutoInplace(context_, input_cipher_k_rot, weight_plain, m_scalingFactorsReal_, m_scalingFactorsRealBig_);

							result.ctks_.push_back(input_cipher_k_rot);
							first_flag = false;
						}
						else
						{

							PhantomCiphertext input_cipher_k = input.ctks_[k];

							PhantomCiphertext input_cipher_k_rot;

							PhantomPlaintext weight_plain = PlainTextEncode(weight_sp, input_cipher_k.chain_index());
							EvalRotateFused(context_, gk_, input_cipher_k, input_cipher_k_rot, rot * pow_slotstr);

							EvalMultAutoInplace(context_, input_cipher_k_rot, weight_plain, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
							EvalAddAutoInplace(context_, result.ctks_[h], input_cipher_k_rot, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}
			}
		}
		result.width_ = (input.width_) / stride;
		result.num_ch_ = out_ch;
		result.slotstr_ = (stride == 2) ? input.slotstr_ + 1 : input.slotstr_;
		Timer::stopGPUTimer("Convolution");
		return result;
	}

	TensorCT DNN::Relu(const TensorCT& input, const PhantomRelinKey& relin_keys, int a, int b, int deg)
	{
		TensorCT result;
		auto relu = [](double x)
			{ return std::max(0.0, x); };

		std::vector<double> coefficients = EvalChebyshevCoefficients(relu, a, b, deg);
		for (int i = 0; i < input.num_ch_; i++)
		{
			result.ctks_.push_back(EvalChebyshevSeries(context_, relin_keys, input.ctks_[i], coefficients, a, b, m_scalingFactorsReal_, m_scalingFactorsRealBig_));
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		return result;
	}

	TensorCT DNN::Sign(const TensorCT& input, const PhantomRelinKey& relin_keys, size_t k)
	{
		TensorCT result;
		std::vector<std::vector<double>> coefficients = { {{0, 0.667972070856, 0, -0.223989523020, 0, 0.136121229346, 0, -0.099160550898, 0, 0.079224867308, 0, -0.067250088206, 0, 0.059852569462, 0, -0.503955481350},
															{0,0.955669291788, 0, -0.317870998995, 0, 0.189953989728, 0, -0.134924463410, 0, 0.104260767625, 0, -0.084798113265, 0, 0.071534728674, 0, -0.282024623439},
															{0, 1.254717353059, 0, -0.371638622338, 0, 0.175181567419, 0, -0.085946606966, 0, 0.039326533561, 0, -0.015616729371, 0, 0.004903749402, 0, -0.000987938705}} };


		double a = (k == 0) ? -1 : (k == 1) ? -1.908 : -1.332;
		double b = (k == 0) ? 1 : (k == 1) ? 1.908 : 1.332;

		for (int i = 0; i < input.num_ch_; i++)
		{
			result.ctks_.push_back(EvalChebyshevSeries(
				context_, relin_keys, input.ctks_[i], coefficients[k], a, b, m_scalingFactorsReal_, m_scalingFactorsRealBig_));
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		return result;
	}

	TensorCT DNN::ReluComposite(const TensorCT& input, FHECKKSRNS& bootstrapper)
	{
		Timer::startGPUTimer("Relu");
		TensorCT result;
		TensorCT sign = input;

		for (int i = 0; i < sign.num_ch_; i++) {
			EvalMultConstInplace(context_, sign.ctks_[i], 0.1, m_scalingFactorsReal_); //Scale Down so it can fall into [-1, 1]
		}
		sign = Sign(sign, mul_key_, 0);
		Timer::stopGPUTimer("Relu");

		Timer::startGPUTimer("Bootstrap");
		for (int i = 0; i < sign.num_ch_; i++) {
			sign.ctks_[i] = bootstrapper.EvalBootstrap(sign.ctks_[i], context_, (input.width_ << input.slotstr_) * (input.width_ << input.slotstr_));
		}
		Timer::stopGPUTimer("Bootstrap");

		Timer::startGPUTimer("Relu");
		sign = Sign(sign, mul_key_, 1);
		Timer::stopGPUTimer("Relu");

		Timer::startGPUTimer("Bootstrap");
		for (int i = 0; i < sign.num_ch_; i++) {
			sign.ctks_[i] = bootstrapper.EvalBootstrap(sign.ctks_[i], context_, (input.width_ << input.slotstr_) * (input.width_ << input.slotstr_));
		}
		Timer::stopGPUTimer("Bootstrap");

		Timer::startGPUTimer("Relu");
		sign = Sign(sign, mul_key_, 2);

		for (int i = 0; i < sign.num_ch_; i++)
		{
			EvalAddConstInPlaceWrap(context_, sign.ctks_[i], 1, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			PhantomCiphertext temp = EvalMultConst(context_, input.ctks_[i], 0.5, m_scalingFactorsReal_);
			result.ctks_.push_back(EvalMultAuto(context_, sign.ctks_[i], temp, mul_key_, m_scalingFactorsReal_, m_scalingFactorsRealBig_));
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		Timer::stopGPUTimer("Relu");

		Timer::startGPUTimer("Bootstrap");
		for (int i = 0; i < result.num_ch_; i++) {
			result.ctks_[i] = bootstrapper.EvalBootstrap(result.ctks_[i], context_, (input.width_ << input.slotstr_) * (input.width_ << input.slotstr_));
		}
		Timer::stopGPUTimer("Bootstrap");

		return result;
	}



	TensorCT DNN::ReluDegTwo(const TensorCT& input, const PhantomRelinKey& relin_keys)
	{
		TensorCT result;
		for (int i = 0; i < input.num_ch_; i++)
		{
			PhantomCiphertext x2_cipher = EvalSquare(context_, input.ctks_[i], relin_keys, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			PhantomCiphertext x_cipher = EvalAddAuto(context_, input.ctks_[i], x2_cipher, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			result.ctks_.push_back(x_cipher);
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		return result;
	}

	TensorCT DNN::BootStrap(TensorCT& input, FHECKKSRNS& bootstrapper)
	{
		TensorCT result;
		for (int i = 0; i < input.num_ch_; i++)
		{
			result.ctks_.push_back(bootstrapper.EvalBootstrap(input.ctks_[i], context_, (input.width_ << input.slotstr_) * (input.width_ << input.slotstr_)));
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		return result;
	}

	void DNN::ComputeRotationIndices(PhantomSecretKey& secret_key, size_t input_width, size_t kernel_h, size_t slotstr)
	{
		std::unordered_set<int32_t> unique_rotations;

		size_t large_L = input_width * std::pow(2, static_cast<int>(slotstr));

		for (int j = 0; j < static_cast<int>(kernel_h); ++j)
		{
			for (int i = 0; i < static_cast<int>(kernel_h); ++i)
			{
				int rot = (i - static_cast<int>(kernel_h / 2)) * static_cast<int>(large_L) + (j - static_cast<int>(kernel_h / 2));

				int rotation_index = rot * std::pow(2, static_cast<int>(slotstr));
				unique_rotations.insert(rotation_index);
			}
		}

		// Convert to vector
		std::vector<int32_t> unique_rotations_idx(unique_rotations.begin(), unique_rotations.end());
		gk_ = secret_key.EvalRotateKeyGen(context_, unique_rotations_idx);
	}

	void DNN::AddRotationIndicesTo(std::vector<int32_t>& rotation_indices, size_t input_width, size_t kernel_h, size_t slotstr)
	{
		std::unordered_set<int32_t> existing(rotation_indices.begin(), rotation_indices.end());
		size_t large_L = input_width * (1 << slotstr);

		for (int j = 0; j < static_cast<int>(kernel_h); ++j)
		{
			for (int i = 0; i < static_cast<int>(kernel_h); ++i)
			{
				int rot = (i - static_cast<int>(kernel_h / 2)) * static_cast<int>(large_L)
					+ (j - static_cast<int>(kernel_h / 2));

				int rotation_index = rot << static_cast<int>(slotstr);

				if (existing.insert(rotation_index).second) {
					// Not already present, add to output
					rotation_indices.push_back(rotation_index);
				}
			}
		}
	}

	void DNN::AddAvgPoolRotationsTo(std::vector<int32_t>& rotation_indices, size_t input_width, size_t slotstr)
	{
		std::unordered_set<int32_t> existing(rotation_indices.begin(), rotation_indices.end());
		int log_l = static_cast<int>(std::log2(input_width));
		int pow_slotstr = 1 << slotstr;

		for (int i = 0; i < log_l; ++i) {
			int rotation = pow_slotstr << i; // 2^slotstr * 2^i
			if (existing.insert(rotation).second) {
				rotation_indices.push_back(rotation);
			}
		}

		for (int j = 0; j < log_l; ++j) {
			int rotation = (pow_slotstr << j) * static_cast<int>(input_width); // 2^slotstr * 2^j * width
			if (existing.insert(rotation).second) {
				rotation_indices.push_back(rotation);
			}
		}
	}


	void DNN::BuildGaloisKey(PhantomSecretKey& secret_key, std::vector<int32_t>& rotation_indices)
	{
		gk_ = secret_key.EvalRotateKeyGen(context_, rotation_indices);
	}


	TensorCT DNN::SoftMax(const TensorCT& input, const PhantomRelinKey& relin_keys, FHECKKSRNS& bootstrapper, size_t BoundB, size_t BoundR, size_t GumbelLambda, size_t GoldSchmidtD)
	{
		TensorCT result;
		auto softmax = [](double x)
			{ return std::exp(x); };

		size_t log_B = std::log2(BoundB);
		size_t log_Lambda = std::log2(GumbelLambda);

		PhantomCiphertext sum_cipher;
		// Degree 12 Chebyshev polynomial
		std::vector<double> coefficients = EvalChebyshevCoefficients(softmax, -1, 1, 12);
		for (int k = 0; k < input.num_ch_; k++)
		{
			result.ctks_.push_back(EvalMultConst(context_, input.ctks_[k], 1 / BoundB, m_scalingFactorsReal_));
			result.ctks_[k] = EvalChebyshevSeries(context_, relin_keys, result.ctks_[k], coefficients, -1, 1, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			for (int i = 0; i < log_B - log_Lambda; i++)
			{
				result.ctks_[k] = EvalSquare(context_, result.ctks_[k], relin_keys, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			}
			sum_cipher = (k == 0) ? result.ctks_[k] : EvalAddAuto(context_, sum_cipher, result.ctks_[k], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}

		sum_cipher = EvalMultConst(context_, sum_cipher, (-1.0) / BoundR, m_scalingFactorsReal_);
		EvalAddConstInPlaceWrap(context_, sum_cipher, 2, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		PhantomCiphertext temp_ct = sum_cipher;
		EvalAddConstInPlaceWrap(context_, temp_ct, -1, m_scalingFactorsReal_, m_scalingFactorsRealBig_);

		for (int j = 0; j < GoldSchmidtD; j++)
		{
			EvalSquareInPlace(context_, temp_ct, relin_keys, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			PhantomCiphertext temp2 = temp_ct;
			EvalAddConstInPlaceWrap(context_, temp2, 1, m_scalingFactorsReal_, m_scalingFactorsRealBig_);

			sum_cipher = EvalMultAuto(context_, sum_cipher, temp2, relin_keys, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}

		for (int k = 0; k < input.num_ch_; k++)
		{
			result.ctks_[k] = EvalMultAuto(context_, result.ctks_[k], sum_cipher, relin_keys, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}

		result.width_ = input.width_;
		result.num_ch_ = input.num_ch_;
		result.slotstr_ = input.slotstr_;
		return result;
	}

	TensorCT DNN::AvgPoolFullCon(const TensorCT& input, const std::vector<std::vector<double>>& weight, const std::vector<double>& bias)
	{
		Timer::startGPUTimer("PoolFC");
		TensorCT tmp_result;
		TensorCT result;

		size_t T = weight.size();	 // Number of rows
		size_t t = weight[0].size(); // Number of columns
		int pow_slotstr = std::pow(2, input.slotstr_);
		size_t log_l = std::log2(input.width_);

		for (int k = 0; k < input.num_ch_; k++)
		{
			tmp_result.ctks_.push_back(input.ctks_[k]);
			for (int i = 0; i < log_l; i++)
			{
				PhantomCiphertext tmp_ct;
				EvalRotateFused(context_, gk_, tmp_result.ctks_[k], tmp_ct, pow_slotstr << i);
				tmp_result.ctks_[k] = EvalAddAuto(context_, tmp_ct, tmp_result.ctks_[k], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			}

			for (int j = 0; j < log_l; j++)
			{
				PhantomCiphertext tmp_ct;
				EvalRotateFused(context_, gk_, tmp_result.ctks_[k], tmp_ct, (pow_slotstr << j) * input.width_);
				tmp_result.ctks_[k] = EvalAddAuto(context_, tmp_ct, tmp_result.ctks_[k], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
			}
		}

		for (int u = 0; u < T; u++)
		{
			for (int k = 0; k < t; k++)
			{
				if (u == 0)
				{
					result.ctks_.push_back(EvalMultConst(context_, tmp_result.ctks_[k], weight[u][k], m_scalingFactorsReal_));
				}

				else
				{
					PhantomCiphertext tmp_ct = EvalMultConst(context_, tmp_result.ctks_[k], weight[u][k], m_scalingFactorsReal_);
					result.ctks_[u] = EvalAddAuto(context_, tmp_ct, result.ctks_[u], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
				}
			}
		}

		for (int u = 0; u < T; u++) {
			EvalAddConstInPlaceWrap(context_, result.ctks_[u], bias[u], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}

		result.width_ = input.width_;
		result.num_ch_ = T;
		result.slotstr_ = input.slotstr_;
		Timer::stopGPUTimer("PoolFC");
		return result;
	}

	TensorCT DNN::BatchNorm(const TensorCT& input,
		const std::vector<double>& weight,
		const std::vector<double>& bias,
		const std::vector<double>& mean,
		const std::vector<double>& var)
	{
		Timer::startGPUTimer("BatchNorm");

		const double eps = 1e-5;
		TensorCT output;
		output.ctks_.resize(input.num_ch_);
		output.width_ = input.width_;
		output.num_ch_ = input.num_ch_;
		output.slotstr_ = input.slotstr_;

		for (size_t c = 0; c < input.num_ch_; ++c)
		{
			double a = weight[c] / std::sqrt(var[c] + eps);
			double b = bias[c] - a * mean[c];

			output.ctks_[c] = EvalMultConst(context_, input.ctks_[c], a, m_scalingFactorsReal_);
			EvalAddConstInPlaceWrap(context_, output.ctks_[c], b, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}
		Timer::stopGPUTimer("BatchNorm");

		return output;
	}

	TensorCT DNN::Add(const TensorCT& a, const TensorCT& b)
	{
		if (a.num_ch_ != b.num_ch_ || a.width_ != b.width_ || a.slotstr_ != b.slotstr_) {
			throw std::runtime_error("TensorCT dimension mismatch in Add()");
		}

		TensorCT result;
		result.num_ch_ = a.num_ch_;
		result.width_ = a.width_;
		result.slotstr_ = a.slotstr_;
		result.ctks_.resize(a.num_ch_);

		for (size_t u = 0; u < a.num_ch_; ++u) {
			result.ctks_[u] = EvalAddAuto(
				context_, a.ctks_[u], b.ctks_[u],
				m_scalingFactorsReal_, m_scalingFactorsRealBig_);
		}

		return result;
	}

	void DNN::RelinKeyGen(PhantomSecretKey& secret_key)
	{
		mul_key_ = secret_key.gen_relinkey(context_);
	}


}
