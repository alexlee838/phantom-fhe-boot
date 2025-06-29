#include "convolution.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom
{
	void PhantomConvolution::zero_pad_encode(const PhantomContext &context, const int f_w, PhantomPublicKey &pk,
											 const vector<vector<double>> &din, vector<PhantomCiphertext> &dout)
	{

		const int p_l = (f_w - 1) / 2;
		const int p_r = (f_w - 1) / 2 + ((f_w - 1) % 2 ? 1 : 0);
		const int din0_sqrt = sqrt(din[0].size());
		const int padded_size_sqrt = din0_sqrt + p_l + p_r;
		const int padded_size = padded_size_sqrt * padded_size_sqrt;
		auto &parms = context.get_context_data(context.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();
		auto coeff_mod_size = parms.coeff_modulus().size();
		auto base_rns = context.gpu_rns_tables().modulus();
		auto numSlots = poly_degree / 2;

		int num_packed;
		int num_packed_inv;
		int row_new;

		num_packed = numSlots / padded_size;
		num_packed_inv = padded_size / numSlots;
		if (num_packed > 0)
			row_new = din.size() / num_packed + ((din.size() % num_packed) ? 1 : 0);
		else
			row_new = din.size() * num_packed_inv;

		vector<double> zero_vector(padded_size_sqrt, 0);
		vector<vector<double>> din_new;
		din_new.resize(row_new, vector<double>(0));

		if (num_packed > 0)
		{
			for (int i = 0; i < row_new; i++)
			{
				for (int j = 0; j < num_packed; j++)
				{
					// Zero padding
					for (int k = 0; k < p_l; k++)
						din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
					if (i * num_packed + j < din.size())
					{
						for (int k = 0; k < din[0].size(); k += din0_sqrt)
						{
							din_new[i].insert(din_new[i].end(), p_l, 0);
							din_new[i].insert(din_new[i].end(), din[i * num_packed + j].begin() + k, din[i * num_packed + j].begin() + k + din0_sqrt);
							din_new[i].insert(din_new[i].end(), p_r, 0);
						}
						for (int k = 0; k < p_l; k++)
							din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
					}
					else
					{
						for (int k = 0; k < din[0].size(); k += din0_sqrt)
						{
							din_new[i].insert(din_new[i].end(), p_l, 0);
							din_new[i].insert(din_new[i].end(), din[din.size() - 1].begin() + k, din[din.size() - 1].begin() + k + din0_sqrt);
							din_new[i].insert(din_new[i].end(), p_r, 0);
						}
						for (int k = 0; k < p_l; k++)
							din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
					}
				}
			}
		}
		else
		{
			int ptr = 0;
			for (int i = 0; i < row_new; i += num_packed_inv)
			{
				ptr = 0;
				for (int j = 0; j < num_packed_inv; j++)
				{
					if (j == 0)
					{
						din_new[i + j].insert(din_new[i + j].end(), zero_vector.begin(), zero_vector.end());
						for (int k = 0; k < (numSlots - padded_size_sqrt); k++)
						{
							if (k % padded_size_sqrt == 0)
								din_new[i + j].insert(din_new[i + j].end(), p_l, 0);
							else if (k % padded_size_sqrt == padded_size_sqrt - 1)
								din_new[i + j].insert(din_new[i + j].end(), p_r, 0);
							else
							{
								din_new[i + j].insert(din_new[i + j].end(), din[i / num_packed_inv].begin() + ptr, din[i / num_packed_inv].begin() + ptr + 1);
								ptr++;
							}
						}
					}
					else if ((j != 0) && (j != num_packed_inv - 1))
					{
						ptr -= 2 * din0_sqrt;
						for (int k = 0; k < (numSlots); k++)
						{
							if (k % padded_size_sqrt == 0)
								din_new[i + j].insert(din_new[i + j].end(), p_l, 0);
							else if (k % padded_size_sqrt == padded_size_sqrt - 1)
								din_new[i + j].insert(din_new[i + j].end(), p_r, 0);
							else
							{
								din_new[i + j].insert(din_new[i + j].end(), din[i / num_packed_inv].begin() + ptr, din[i / num_packed_inv].begin() + ptr + 1);
								ptr++;
							}
						}
					}
					else
					{
						ptr -= 2 * din0_sqrt;
						for (int k = 0; k < (numSlots - padded_size_sqrt); k++)
						{
							if (k % padded_size_sqrt == 0)
								din_new[i + j].insert(din_new[i + j].end(), p_l, 0);
							else if (k % padded_size_sqrt == padded_size_sqrt - 1)
								din_new[i + j].insert(din_new[i + j].end(), p_r, 0);
							else
							{
								din_new[i + j].insert(din_new[i + j].end(), din[i / num_packed_inv].begin() + ptr, din[i / num_packed_inv].begin() + ptr + 1);
								ptr++;
							}
						}
						din_new[i + j].insert(din_new[i + j].end(), zero_vector.begin(), zero_vector.end());
					}
				}
			}
		}

		PhantomCKKSEncoder encoder(context);

		for (int i = 0; i < row_new; i++)
		{
			PhantomPlaintext temp;
			PhantomCiphertext temp_cipher;
			encoder.encode(context, din_new[i], scale_, temp);
			pk.encrypt_asymmetric(context, temp, temp_cipher);

			dout.emplace_back(temp_cipher);
		}

		dout[0].PreComputeScale(context, scale_);
		m_scalingFactorsReal_ = dout[0].getScalingFactorsReal();
		m_scalingFactorsRealBig_ = dout[0].getScalingFactorsRealBig();

		std::cout << "Message size: " << 32 * dout.size() << " KB" << endl;
	}

	void PhantomConvolution::EvalConvolution(const int in_h, vector<vector<vector<double>>> &filter, const PhantomContext &context,
											 const PhantomGaloisKey &gal_keys, const vector<PhantomCiphertext> &din, vector<PhantomCiphertext> &dout)
	{

		int num_fout = filter.size();
		int num_in = filter[0].size();
		int f_size = filter[0][0].size();
		int f_h = sqrt(f_size);

		int p_l = (f_h - 1) / 2;
		int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
		int wPp = in_h + p_l + p_r;
		int in_size = wPp * wPp;
		auto &parms = context.get_context_data(context.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();
		int numSlots = poly_degree / 2;

		int n_i;
		std::cout << f_size << " " << num_in << " " << num_fout << " " << std::endl;
		int num_out = f_size * num_in * num_fout;
		int num_out2 = 0;
		if (in_size > numSlots)
		{
			num_out2 = (f_h - 1) * f_h * num_in * num_fout;
			n_i = in_size / numSlots;
		}
		else
		{
			num_out = f_size * num_in * (num_fout / (numSlots / in_size) + (num_fout % (numSlots / in_size) ? 1 : 0));
			n_i = numSlots / in_size;
		}

		std::cout << "num_fout: " << num_fout << '\n';
		std::cout << "n_i: " << n_i << '\n';
		std::cout << "num_out: " << num_out << '\n';
		std::cout << "num_out2: " << num_out2 << '\n';

		int num_ct_in = din.size();
		int num_ct_out;
		if (in_size > numSlots)
			num_ct_out = num_fout * n_i;
		else
			num_ct_out = num_fout / n_i + ((num_fout % n_i) ? 1 : 0);

		dout.reserve(num_ct_out);

		std::cout << "num_ct_out: " << num_ct_out << '\n';

		// Construct Hadamard matrix and NTT of the matrix
		vector<vector<int>> hadamard_n_i(n_i, vector<int>(n_i));

		hadamard_n_i[0][0] = 1;
		for (int k = 1; k < n_i; k += k)
		{
			for (int i = 0; i < k; i++)
			{
				for (int j = 0; j < k; j++)
				{
					hadamard_n_i[i + k][j] = hadamard_n_i[i][j];
					hadamard_n_i[i][j + k] = hadamard_n_i[i][j];
					hadamard_n_i[i + k][j + k] = -hadamard_n_i[i][j];
				}
			}
		}

		vector<PhantomPlaintext> hadamard_ntt(n_i, PhantomPlaintext());
		vector<double> hadamard_vec(numSlots, 0.0);

		PhantomCKKSEncoder encoder(context);

		vector<double> weight(numSlots);

		double p_1 = 1 / (double)n_i;
		double m_1 = -1 / (double)n_i;

		for (int i = 0; i < n_i; i++)
		{
			if (i != 0)
			{
				for (int j = 0; j < n_i; j++)
				{
					if (hadamard_n_i[i][j] != 1)
					{
						for (int k = (j * (numSlots / n_i)); k < ((j + 1) * (numSlots / n_i)); k++)
						{
							hadamard_vec[k] = m_1;
						}
					}
					else
					{
						for (int k = (j * (numSlots / n_i)); k < ((j + 1) * (numSlots / n_i)); k++)
						{
							hadamard_vec[k] = p_1;
						}
					}
				}
				encoder.encode(context, hadamard_vec, scale_, hadamard_ntt[i]);
			}
			else
			{
				for (int i = 0; i < numSlots; i++)
				{
					hadamard_vec[i] = p_1;
				}
				encoder.encode(context, hadamard_vec, scale_, hadamard_ntt[0]);
			}
		}

		// Transform a filter to have power of 2 index
		int c_o_round = num_fout;
		int c_i_round = num_in;

		while (c_o_round % n_i != 0)
		{
			c_o_round++;
		}
		while (c_i_round % n_i != 0)
		{
			c_i_round++;
		}

		// Construct filter encoded with hadamard and filter_round
		std::cout << "num_ct_in: " << num_ct_in << "\n";

		double partial_s;

		vector<vector<vector<double>>> filter_round(c_o_round, vector<vector<double>>(c_i_round, vector<double>(f_size, 0)));
		vector<vector<vector<vector<vector<double>>>>> enc_filter_round(num_ct_in, vector<vector<vector<vector<double>>>>(n_i, vector<vector<vector<double>>>(num_ct_out, vector<vector<double>>(f_size, vector<double>(n_i, 0)))));

		for (int i = 0; i < num_fout; i++)
		{
			for (int j = 0; j < num_in; j++)
			{
				for (int k = 0; k < f_size; k++)
					filter_round[i][j][k] = filter[i][j][k];
			}
		}
		for (int i = 0; i < num_ct_in; i++)
		{
			for (int j = 0; j < (num_ct_out); j++)
			{
				for (int k = 0; k < n_i; k++)
				{
					for (int m = 0; m < n_i; m++)
					{
						for (int idx = 0; idx < f_size; idx++)
						{
							partial_s = 0;
							for (int l = 0; l < n_i; l++)
							{
								if (n_i != 1)
								{
									partial_s += (filter_round[((l + k) % n_i) + j * n_i][i * n_i + l][idx] * hadamard_n_i[m][l]);
								}
								else
								{
									partial_s += (filter_round[(l + k) + j * n_i][i * n_i + l][idx] * hadamard_n_i[m][l]);
								}
							}

							enc_filter_round[i][k][j][idx][m] = partial_s;
						}
					}
				}
			}
		}

		dout.resize(num_ct_out);

		if (in_size <= numSlots)
		{
			if (n_i != 2)
			{
				vector<vector<PhantomCiphertext>> tmp_out(num_ct_out, vector<PhantomCiphertext>(n_i, din[0]));

				for (int i = 0; i < num_ct_in; i++)
				{
					ConvolutionOP(wPp, context, enc_filter_round[i], din[i], hadamard_ntt, tmp_out);

					for (int j = 0; j < n_i; j++)
					{
						if (n_i != 1)
						{
							for (int k = 0; k < num_ct_out; k++)
							{
								rotate_inplace(context, tmp_out[k][j], -j * in_size, gal_keys);
							}
						}
						if ((i == 0) && (j == 0))
						{
							for (int k = 0; k < num_ct_out; k++)
							{
								dout[k] = (tmp_out[k][j]);
							}
						}
						else
						{
							for (int k = 0; k < num_ct_out; k++)
							{
								EvalAddAutoInplace(context, dout[k], tmp_out[k][j], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
							}
						}
					}
				}
			}

			else
			{
				vector<vector<vector<PhantomCiphertext>>> tmp_out2(num_ct_in, vector<vector<PhantomCiphertext>>(num_ct_out, vector<PhantomCiphertext>(n_i, din[0])));

				for (int i = 0; i < num_ct_in; i++)
				{
					ConvolutionOP(wPp, context, enc_filter_round[i], din[i], hadamard_ntt, tmp_out2[i]);
				}

				for (int i = 0; i < num_ct_in; i++)
				{
					for (int k = 0; k < num_ct_out; k++)
					{
						if (i != 0)
						{
							EvalAddAutoInplace(context, tmp_out2[0][k][0], tmp_out2[i][k][0], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
							EvalAddAutoInplace(context, tmp_out2[0][k][1], tmp_out2[i][k][1], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}
				for (int k = 0; k < num_ct_out; k++)
				{
					rotate_inplace(context, tmp_out2[0][k][1], numSlots / 2, gal_keys);
					EvalAddAutoInplace(context, tmp_out2[0][k][0], tmp_out2[0][k][1], m_scalingFactorsReal_, m_scalingFactorsRealBig_);

					dout[k] = tmp_out2[0][k][0];
				}
			}
		}
		else
		{ // Not fixed // Another function
			throw logic_error("Not yet to support n < (d_size_sqrt + p_l + p_r)**2");
		}

		std::cout << "Message size: " << 32 * dout.size() << " KB" << endl;
	}

	

	void PhantomConvolution::SetRotationKeys(const PhantomContext &context, PhantomSecretKey &secret_key, const int in_h, const int f_h)
	{
		std::unordered_set<int32_t> rotation_set;

		int32_t half_f_h = (f_h - 1) / 2;
		int p_l = (f_h - 1) / 2;
		int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
		int wPp = in_h + p_l + p_r;

		// First set of rotations (from `i` loop)
		for (int32_t i = 1; i <= half_f_h; i++)
		{
			rotation_set.insert(i);
			rotation_set.insert(-i);
			rotation_set.insert(wPp * i);
			rotation_set.insert(-wPp * i);
		}

		// Second set of rotations (from `j` loop)
		for (int32_t i = 1; i <= half_f_h; i++)
		{
			for (int32_t j = 1; j <= half_f_h; j++)
			{
				rotation_set.insert(j + wPp * i);
				rotation_set.insert(-j + wPp * i);
				rotation_set.insert(j - wPp * i);
				rotation_set.insert(-j - wPp * i);
			}
		}

		// Convert the unordered_set to a vector
		std::vector<int32_t> rotation_indices(rotation_set.begin(), rotation_set.end());

		galois_keys_ = secret_key.EvalRotateKeyGen(context, rotation_indices);
	}

	// template <int N>
	void PhantomConvolution::ConvolutionOP(const int in_h, const PhantomContext &context, vector<vector<vector<vector<double>>>> &enc_filter,
										   const PhantomCiphertext &din, const vector<PhantomPlaintext> &hadamard_ntt, vector<vector<PhantomCiphertext>> &dout)
	{
		int num_ct_out = enc_filter[0].size();
		int f_size = enc_filter[0][0].size();
		int n_i = enc_filter[0][0][0].size();
		int f_h = sqrt(f_size);

		auto &parms = context.get_context_data(context.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();

		dout.resize(num_ct_out, vector<PhantomCiphertext>(n_i));

		vector<PhantomCiphertext> tmp(8);

		auto digits = EvalFastRotationPrecompute(context, din);
		vector<vector<vector<PhantomCiphertext>>> result3(
			n_i, std::vector<std::vector<PhantomCiphertext>>(
					 num_ct_out, std::vector<PhantomCiphertext>(n_i)));

		// new
		for (int i = 0; i < num_ct_out; i++)
		{
			for (int j = 0; j < n_i; j++)

			{
				for (int k = 0; k < n_i; k++)
				{
					result3[j][i][k] = EvalMultConst(context, din, enc_filter[j][i][(f_size - 1) / 2][k], m_scalingFactorsReal_);
				}
			}
		}

		for (int i = 1; i < (f_h - 1) / 2 + 1; i++)
		{

			tmp[0] = EvalFastRotationExt(context, din, galois_keys_, i, digits, true);
			tmp[0] = KeySwitchDown(context, tmp[0]);

			for (int j = 0; j < num_ct_out; j++)
			{

				for (int m = 0; m < n_i; m++)
				{

					for (int l = 0; l < n_i; l++)
					{
						auto const_ct_temp = EvalMultConst(context, tmp[0], enc_filter[m][j][(f_size - 1) / 2 + i][l], m_scalingFactorsReal_);

						EvalAddAutoInplace(context, result3[m][j][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
					}
				}
			}

			tmp[1] = EvalFastRotationExt(context, din, galois_keys_, -i, digits, true);
			tmp[1] = KeySwitchDown(context, tmp[1]);

			for (int j = 0; j < num_ct_out; j++)
			{
				for (int m = 0; m < n_i; m++)
				{

					for (int l = 0; l < n_i; l++)
					{
						auto const_ct_temp = EvalMultConst(context, tmp[1], enc_filter[m][j][(f_size - 1) / 2 - i][l], m_scalingFactorsReal_);
						EvalAddAutoInplace(context, result3[m][j][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
					}
				}
			}

			tmp[2] = EvalFastRotationExt(context, din, galois_keys_, in_h * i, digits, true);
			tmp[2] = KeySwitchDown(context, tmp[2]);

			for (int j = 0; j < num_ct_out; j++)
			{
				for (int m = 0; m < n_i; m++)
				{

					for (int l = 0; l < n_i; l++)
					{
						auto const_ct_temp = EvalMultConst(context, tmp[2], enc_filter[m][j][(f_size - 1) / 2 + i * f_h][l], m_scalingFactorsReal_);
						EvalAddAutoInplace(context, result3[m][j][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
					}
				}
			}
			tmp[3] = EvalFastRotationExt(context, din, galois_keys_, -in_h * i, digits, true);
			tmp[3] = KeySwitchDown(context, tmp[3]);

			for (int j = 0; j < num_ct_out; j++)
			{
				for (int m = 0; m < n_i; m++)
				{

					for (int l = 0; l < n_i; l++)
					{
						auto const_ct_temp = EvalMultConst(context, tmp[3], enc_filter[m][j][(f_size - 1) / 2 - i * f_h][l], m_scalingFactorsReal_);
						EvalAddAutoInplace(context, result3[m][j][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
					}
				}
			}

			for (int j = 1; j < (f_h - 1) / 2 + 1; j++)
			{
				tmp[4] = EvalFastRotationExt(context, din, galois_keys_, j + in_h * i, digits, true);
				tmp[4] = KeySwitchDown(context, tmp[4]);

				for (int k = 0; k < num_ct_out; k++)
				{
					for (int m = 0; m < n_i; m++)
					{

						for (int l = 0; l < n_i; l++)
						{
							auto const_ct_temp = EvalMultConst(context, tmp[4], enc_filter[m][k][(f_size - 1) / 2 + i * f_h + j][l], m_scalingFactorsReal_);
							EvalAddAutoInplace(context, result3[m][k][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}
				tmp[5] = EvalFastRotationExt(context, din, galois_keys_, j - in_h * i, digits, true);
				tmp[5] = KeySwitchDown(context, tmp[5]);

				for (int k = 0; k < num_ct_out; k++)
				{
					for (int m = 0; m < n_i; m++)
					{

						for (int l = 0; l < n_i; l++)
						{
							auto const_ct_temp = EvalMultConst(context, tmp[5], enc_filter[m][k][(f_size - 1) / 2 - i * f_h + j][l], m_scalingFactorsReal_);
							EvalAddAutoInplace(context, result3[m][k][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}

				tmp[6] = EvalFastRotationExt(context, din, galois_keys_, -j + in_h * i, digits, true);
				tmp[6] = KeySwitchDown(context, tmp[6]);

				for (int k = 0; k < num_ct_out; k++)
				{
					for (int m = 0; m < n_i; m++)
					{

						for (int l = 0; l < n_i; l++)
						{
							auto const_ct_temp = EvalMultConst(context, tmp[6], enc_filter[m][k][(f_size - 1) / 2 + i * f_h - j][l], m_scalingFactorsReal_);
							EvalAddAutoInplace(context, result3[m][k][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}

				tmp[7] = EvalFastRotationExt(context, din, galois_keys_, -j - in_h * i, digits, true);
				tmp[7] = KeySwitchDown(context, tmp[7]);

				for (int k = 0; k < num_ct_out; k++)
				{
					for (int m = 0; m < n_i; m++)
					{

						for (int l = 0; l < n_i; l++)
						{
							auto const_ct_temp = EvalMultConst(context, tmp[7], enc_filter[m][k][(f_size - 1) / 2 - i * f_h - j][l], m_scalingFactorsReal_);
							EvalAddAutoInplace(context, result3[m][k][l], const_ct_temp, m_scalingFactorsReal_, m_scalingFactorsRealBig_);
						}
					}
				}
			}
		}
		for (int i = 0; i < num_ct_out; i++)
		{
			for (int j = 0; j < n_i; j++)
			{
				for (int m = 0; m < n_i; m++)
				{
					EvalMultAutoInplace(context, result3[m][i][j], hadamard_ntt[j], m_scalingFactorsReal_, m_scalingFactorsRealBig_);

					if (j != 0)
						EvalAddAutoInplace(context, dout[i][m], result3[m][i][j], m_scalingFactorsReal_, m_scalingFactorsRealBig_);
					else
						dout[i][m] = result3[m][i][j];
				}
			}
		}
	}

	void PhantomConvolution::ConvDecode(const PhantomContext &context, PhantomSecretKey &secret_key, const vector<PhantomCiphertext> &dout, vector<vector<double>> &img_out)
	{
		PhantomCKKSEncoder encoder(context);
		int num_of_ciphers = dout.size();
		img_out.resize(num_of_ciphers, vector<double>(0));
		vector<double> temp_result;
		PhantomPlaintext temp_plain;

		for (int i = 0; i < num_of_ciphers; i++)
		{
			secret_key.decrypt(context, dout[i], temp_plain);
			encoder.decode(context, temp_plain, temp_result);

			img_out[i].insert(img_out[i].end(), temp_result.begin(), temp_result.end());
		}
	}

	std::vector<PhantomPlaintext> PhantomConvolution::FCWeightEncodeCore(const PhantomContext &context, const std::vector<std::vector<double>> &weight)
	{

		auto &parms = context.get_context_data(context.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();
		PhantomCKKSEncoder encoder(context);

		// Row and Col are each power of 2
		// FCWeightEncodeCore assumes row & col <= numSlots
		size_t row = weight.size();
		size_t col = weight[0].size();

		std::vector<double> DiagonalWeight(col, 0.0);

		std::vector<PhantomPlaintext> DiagonalWeightPlain(row);

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				DiagonalWeight[j] = weight[(i + j) % row][j];
			}
			encoder.encode(context, DiagonalWeight, scale_, DiagonalWeightPlain[i]);
		}
		return DiagonalWeightPlain;
	}

	PhantomPlaintext PhantomConvolution::FCBiasEncodeCore(const PhantomContext &context, const std::vector<double> &bias)
	{
		PhantomCKKSEncoder encoder(context);

		auto &parms = context.get_context_data(context.get_first_index()).parms();
		PhantomPlaintext bias_plain;
		encoder.encode(context, bias, scale_, bias_plain);

		return bias_plain;
	}

	PhantomCiphertext PhantomConvolution::FullyConnectedLayerCore(const PhantomContext &context, const PhantomGaloisKey &gal_keys, const PhantomCiphertext &din,
																  const std::vector<PhantomPlaintext> &weight, const PhantomPlaintext &bias, const size_t col, bool add_bias)
	{
		PhantomCiphertext prev = din;

		multiply_plain_inplace(context, prev, weight[0]);
		rescale_to_next_inplace(context, prev);

		auto &parms = context.get_context_data(context.get_first_index()).parms();
		auto poly_degree = parms.poly_modulus_degree();
		auto numSlots = poly_degree / 2;

		bool doubleRotate = true;
		if (col == numSlots)
		{
			doubleRotate = false;
		}

		for (int i = 1; i < weight.size(); i++)
		{
			PhantomCiphertext tmp = din;
			multiply_plain_inplace(context, tmp, weight[i]);
			rescale_to_next_inplace(context, tmp);

			if (doubleRotate)
			{
				PhantomCiphertext tmp2 = tmp;
				rotate_inplace(context, tmp2, col - i, gal_keys);
				add_inplace(context, prev, tmp2);
			}

			rotate_inplace(context, tmp, -i, gal_keys);
			add_inplace(context, prev, tmp);
		}

		PhantomCiphertext dout = prev;
		for (int i = col / 2; i >= weight.size(); i /= 2)
		{
			rotate_inplace(context, prev, i, gal_keys);
			add_inplace(context, dout, prev);
			prev = dout;
		}

		if (add_bias)
		{
			add_plain_inplace(context, dout, bias);
		}

		return dout;
	}

void PhantomConvolution::processFullyConnectedLayer(
    const PhantomContext& context,
    const PhantomGaloisKey& galois_keys,
    const std::vector<std::vector<double>>& weight,
    const std::vector<double>& bias,
    const std::vector<PhantomCiphertext>& ct,
    std::vector<PhantomCiphertext>& ct_out,
    int num_of_cipher_in,
    int n_o,
    int n_i)
{
    int element_per_cipher = n_i / num_of_cipher_in; // Ensure divisibility before calling

    if (num_of_cipher_in != 1)
    {
        // Reshape weight into 3D tensor
        std::vector<std::vector<std::vector<double>>> reshaped(num_of_cipher_in,
            std::vector<std::vector<double>>(n_o, std::vector<double>(element_per_cipher)));

        for (int i = 0; i < n_o; ++i)
        {
            for (int j = 0; j < n_i; ++j)
            {
                int c = j / element_per_cipher;
                int new_j = j % element_per_cipher;
                reshaped[c][i][new_j] = weight[i][j];
            }
        }

        // Encode reshaped weights
        std::vector<std::vector<PhantomPlaintext>> DiagWeightPlain(num_of_cipher_in, std::vector<PhantomPlaintext>(n_o));
        for (int i = 0; i < num_of_cipher_in; i++)
        {
            DiagWeightPlain[i] = FCWeightEncodeCore(context, reshaped[i]);
        }

        // Encode bias
        auto BiasPlain = FCBiasEncodeCore(context, bias);

        // Perform the first computation
        ct_out[0] = FullyConnectedLayerCore(context, galois_keys, ct[0], DiagWeightPlain[0], BiasPlain, element_per_cipher, true);

        // Accumulate results
        for (int i = 1; i < num_of_cipher_in; i++)
        {
            auto temp = FullyConnectedLayerCore(context, galois_keys, ct[i], DiagWeightPlain[i], BiasPlain, element_per_cipher, false);
            add_inplace(context, ct_out[0], temp);
        }
    }
    else
    {
        auto DiagWeightPlain = FCWeightEncodeCore(context, weight);
        auto BiasPlain = FCBiasEncodeCore(context, bias);

        ct_out[0] = FullyConnectedLayerCore(context, galois_keys, ct[0], DiagWeightPlain, BiasPlain, weight[0].size());
    }
}

}
