#include "bootstrap.cuh"

#include "evaluate.cuh"
#include "rns_bconv.cuh"
#include "scalingvariant.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom
{

	void FHECKKSRNS::EvalBootstrapSetup(const PhantomContext &cc, std::vector<uint32_t> &levelBudget, const double &scale,
										const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig,
										std::vector<uint32_t> dim1, uint32_t numSlots, uint32_t correctionFactor,
										bool precompute)
	{

		this->m_scalingFactorsReal = m_scalingFactorsReal;
		this->m_scalingFactorsRealBig = m_scalingFactorsRealBig;

		auto &context_data = cc.get_context_data(cc.get_first_index());
		auto &parms = context_data.parms();
		size_t poly_degree = parms.poly_modulus_degree();

		int M = 2 * poly_degree;
		int slots = (numSlots == 0) ? poly_degree / 2 : numSlots;

		if (correctionFactor == 0)
		{
			auto tmp = std::round(-0.265 * (2 * std::log2(M / 2) + std::log2(slots)) + 19.1);
			if (tmp < 7)
				m_correctionFactor = 7;
			else if (tmp > 13)
				m_correctionFactor = 13;
			else
				m_correctionFactor = static_cast<int>(tmp);
		}

		else
		{
			m_correctionFactor = correctionFactor;
		}

		m_bootPrecomMap[slots] = std::make_shared<CKKSBootstrapPrecom>();
		std::shared_ptr<CKKSBootstrapPrecom> precom = m_bootPrecomMap[slots];

		precom->m_slots = slots;
		precom->m_dim1 = dim1[0];

		uint32_t logSlots = std::log2(slots);

		if (logSlots == 0)
		{
			logSlots = 1;
		}

		std::vector<uint32_t> newBudget = levelBudget;

		if (newBudget[0] > logSlots)
		{
			std::cerr << "\nWarning, the level budget for encoding is too large. Setting it to " << logSlots << std::endl;
			newBudget[0] = logSlots;
		}
		if (newBudget[0] < 1)
		{
			std::cerr << "\nWarning, the level budget for encoding can not be zero. Setting it to 1" << std::endl;
			newBudget[0] = 1;
		}

		if (newBudget[1] > logSlots)
		{
			std::cerr << "\nWarning, the level budget for decoding is too large. Setting it to " << logSlots << std::endl;
			newBudget[1] = logSlots;
		}
		if (newBudget[1] < 1)
		{
			std::cerr << "\nWarning, the level budget for decoding can not be zero. Setting it to 1" << std::endl;
			newBudget[1] = 1;
		}

		precom->m_paramsEnc = GetCollapsedFFTParams(slots, newBudget[0], dim1[0]);
		precom->m_paramsDec = GetCollapsedFFTParams(slots, newBudget[1], dim1[1]);

		if (precompute)
		{
			uint32_t m = 4 * slots;
			bool isSparse = (M != m) ? true : false;

			std::vector<uint32_t> rotGroup(slots);
			uint32_t fivePows = 1;
			for (uint32_t i = 0; i < slots; i++)
			{
				rotGroup[i] = fivePows;
				fivePows *= 5;
				fivePows %= m;
			}

			std::vector<std::complex<double>> ksiPows(m + 1);
			for (uint32_t j = 0; j < m; j++)
			{
				double angle = 2.0 * M_PI * j / m;
				ksiPows[j].real(cos(angle));
				ksiPows[j].imag(sin(angle));
			}
			ksiPows[m] = ksiPows[0];

			double qDouble = static_cast<double>(parms.coeff_modulus()[0].value());

			unsigned __int128 const_one = 1;
			unsigned __int128 factor = const_one << ((uint32_t)std::round(std::log2(qDouble)));
			double pre = qDouble / factor;
			double k = 1.0;
			double scaleEnc = pre / k;
			double scaleDec = 1 / pre;

			uint32_t approxModDepth = GetModDepthInternal();
			uint32_t depthBT = approxModDepth + precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] +
							   precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];

			// compute # of levels to remain when encoding the coefficients
			// Extract encryption parameters.

			auto &key_context_data = cc.get_context_data(0);
			auto &key_parms = key_context_data.parms();
			auto scheme = key_parms.scheme();
			auto &key_modulus = key_parms.coeff_modulus();
			size_t size_P = key_parms.special_modulus_size();
			size_t size_QP = key_modulus.size();

			uint32_t L0 = size_QP - size_P;

			uint32_t lEnc = L0 - precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] - 1;
			uint32_t lDec = L0 - depthBT;

			bool isLTBootstrap = (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) &&
								 (precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);

			if (isLTBootstrap)
			{
				// allocate all vectors
				std::vector<std::vector<std::complex<double>>> U0(slots, std::vector<std::complex<double>>(slots));
				std::vector<std::vector<std::complex<double>>> U1(slots, std::vector<std::complex<double>>(slots));
				std::vector<std::vector<std::complex<double>>> U0hatT(slots, std::vector<std::complex<double>>(slots));
				std::vector<std::vector<std::complex<double>>> U1hatT(slots, std::vector<std::complex<double>>(slots));

				for (size_t i = 0; i < slots; i++)
				{
					for (size_t j = 0; j < slots; j++)
					{
						U0[i][j] = ksiPows[(j * rotGroup[i]) % m];
						U0hatT[j][i] = std::conj(U0[i][j]);
						U1[i][j] = std::complex<double>(0, 1) * U0[i][j];
						U1hatT[j][i] = std::conj(U1[i][j]);
					}
				}

				if (!isSparse)
				{
					// precom->m_U0hatTPre = EvalLinearTransformPrecompute(cc, U0hatT, scaleEnc, lEnc);
					// precom->m_U0Pre     = EvalLinearTransformPrecompute(cc, U0, scaleDec, lDec);
				}
				else
				{
					// precom->m_U0hatTPre = EvalLinearTransformPrecompute(cc, U0hatT, U1hatT, 0, scaleEnc, lEnc);
					// precom->m_U0Pre     = EvalLinearTransformPrecompute(cc, U0, U1, 1, scaleDec, lDec);
				}
			}
			else
			{
				precom->m_U0hatTPreFFT = EvalCoeffsToSlotsPrecompute(cc, ksiPows, rotGroup, this->m_scalingFactorsReal, false, scaleEnc, lEnc);
				precom->m_U0PreFFT = EvalSlotsToCoeffsPrecompute(cc, ksiPows, rotGroup, this->m_scalingFactorsReal, false, scaleDec, lDec);
			}
		}
	}

	//------------------------------------------------------------------------------
	// Precomputations for CoeffsToSlots and SlotsToCoeffs
	//------------------------------------------------------------------------------

	std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> FHECKKSRNS::EvalCoeffsToSlotsPrecompute(
		const PhantomContext &cc, const std::vector<std::complex<double>> &A, const std::vector<uint32_t> &rotGroup, const std::vector<double> &m_scalingFactorsReal,
		bool flag_i, double scale, uint32_t L) const
	{

		uint32_t slots = rotGroup.size();

		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;
		auto &context_data = cc.get_context_data(cc.get_first_index());
		auto &parms = context_data.parms();
		size_t poly_degree = parms.poly_modulus_degree();

		int M = 2 * poly_degree;

		int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t stop = -1;
		int32_t flagRem = 0;

		if (remCollapse != 0)
		{
			stop = 0;
			flagRem = 1;
		}

		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> result(levelBudget);

		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			size_t size = (flagRem == 1 && i == 0) ? numRotationsRem : numRotations;

			// Resize the inner vector and initialize each element
			result[i] = std::vector<std::shared_ptr<PhantomPlaintext>>(size);

			for (size_t j = 0; j < size; j++)
			{
				result[i][j] = std::make_shared<PhantomPlaintext>(); // Allocate a valid object
			}
		}

		// make sure the plaintext is created only with the necessary amount of moduli

		uint32_t towersToDrop = 0;
		uint32_t chain_idx = 1;
		if (L != 0)
		{
			towersToDrop = parms.coeff_modulus().size() - L - levelBudget;
		}

		for (uint32_t i = 0; i < towersToDrop; i++)
		{
			// coeff_mod.pop_back();
			chain_idx++;
		}

		uint32_t level0 = towersToDrop + levelBudget - 1;

		auto &key_context_data = cc.get_context_data(0);

		// we need to pre-compute the plaintexts in the extended basis P*Q
		std::vector<uint32_t> chainVector(levelBudget - stop);
		// std::vector<std::shared_ptr<PhantomContext>> contextVector(levelBudget - stop);
		for (int32_t s = levelBudget - 1; s >= stop; s--)
		{

			chainVector[s - stop] = chain_idx;
			chain_idx++;
		}

		if (slots == M / 4)
		{
			//------------------------------------------------------------------------------
			// fully-packed mode
			//------------------------------------------------------------------------------

			auto coeff = CoeffEncodingCollapse(A, rotGroup, levelBudget, flag_i);

			for (int32_t s = levelBudget - 1; s > stop; s--)
			{
				for (int32_t i = 0; i < b; i++)
				{
					// #pragma omp parallel for
					for (int32_t j = 0; j < g; j++)
					{
						if (g * i + j != int32_t(numRotations))
						{
							uint32_t rot =
								ReduceRotation(-g * i * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots);
							if ((flagRem == 0) && (s == stop + 1))
							{
								// do the scaling only at the last set of coefficients
								for (uint32_t k = 0; k < slots; k++)
								{
									coeff[s][g * i + j][k] *= scale;
								}
							}

							std::vector<cuDoubleComplex> rotateTemp = Rotate(coeff[s][g * i + j], rot);

							encoder_.encode_ext(cc, rotateTemp, m_scalingFactorsReal[level0 - s], *result[s][g * i + j], chainVector[s - stop]);
						}
					}
				}
			}

			if (flagRem)
			{
				for (int32_t i = 0; i < bRem; i++)
				{
					// #pragma omp parallel for
					for (int32_t j = 0; j < gRem; j++)
					{
						if (gRem * i + j != int32_t(numRotationsRem))
						{
							uint32_t rot = ReduceRotation(-gRem * i, slots);
							for (uint32_t k = 0; k < slots; k++)
							{
								coeff[stop][gRem * i + j][k] *= scale;
							}

							auto rotateTemp = Rotate(coeff[stop][gRem * i + j], rot);
							encoder_.encode_ext(cc, rotateTemp, m_scalingFactorsReal[level0], *result[stop][gRem * i + j], chainVector[0]);
						}
					}
				}
			}
		}
		else
		{
			//------------------------------------------------------------------------------
			// sparsely-packed mode
			//------------------------------------------------------------------------------

			auto coeff = CoeffEncodingCollapse(A, rotGroup, levelBudget, false);
			auto coeffi = CoeffEncodingCollapse(A, rotGroup, levelBudget, true);

			for (int32_t s = levelBudget - 1; s > stop; s--)
			{
				for (int32_t i = 0; i < b; i++)
				{
					// #pragma omp parallel for
					for (int32_t j = 0; j < g; j++)
					{
						if (g * i + j != int32_t(numRotations))
						{
							uint32_t rot =
								ReduceRotation(-g * i * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
							// concatenate the coefficients horizontally on their third dimension, which corresponds to the # of slots
							auto clearTemp = coeff[s][g * i + j];
							auto clearTempi = coeffi[s][g * i + j];
							clearTemp.insert(clearTemp.end(), clearTempi.begin(), clearTempi.end());
							if ((flagRem == 0) && (s == stop + 1))
							{
								// do the scaling only at the last set of coefficients
								for (uint32_t k = 0; k < clearTemp.size(); k++)
								{
									clearTemp[k] *= scale;
								}
							}

							auto rotateTemp = Rotate(clearTemp, rot);
							encoder_.encode_sparse_ext(cc, rotateTemp, m_scalingFactorsReal[level0 - s], *result[s][g * i + j], chainVector[s - stop]);
						}
					}
				}
			}

			if (flagRem)
			{
				for (int32_t i = 0; i < bRem; i++)
				{
					// #pragma omp parallel for
					for (int32_t j = 0; j < gRem; j++)
					{
						if (gRem * i + j != int32_t(numRotationsRem))
						{
							uint32_t rot = ReduceRotation(-gRem * i, M / 4);
							// concatenate the coefficients on their third dimension, which corresponds to the # of slots
							auto clearTemp = coeff[stop][gRem * i + j];
							auto clearTempi = coeffi[stop][gRem * i + j];
							clearTemp.insert(clearTemp.end(), clearTempi.begin(), clearTempi.end());
							for (uint32_t k = 0; k < clearTemp.size(); k++)
							{
								clearTemp[k] *= scale;
							}

							auto rotateTemp = Rotate(clearTemp, rot);
							encoder_.encode_sparse_ext(cc, rotateTemp, m_scalingFactorsReal[level0], *result[stop][gRem * i + j], chainVector[0]);
						}
					}
				}
			}
		}

		return result;
	}

	std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> FHECKKSRNS::EvalSlotsToCoeffsPrecompute(
		const PhantomContext &cc, const std::vector<std::complex<double>> &A, const std::vector<uint32_t> &rotGroup, const std::vector<double> &m_scalingFactorsReal,
		bool flag_i, double scale, uint32_t L) const
	{

		uint32_t slots = rotGroup.size();

		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		auto &context_data = cc.get_context_data(cc.get_first_index());
		auto &parms = context_data.parms();
		size_t poly_degree = parms.poly_modulus_degree();

		int M = 2 * poly_degree;

		int32_t levelBudget = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t flagRem = 0;

		if (remCollapse != 0)
		{
			flagRem = 1;
		}

		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> result(levelBudget);

		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			size_t size = (flagRem == 1 && i == uint32_t(levelBudget - 1)) ? numRotationsRem : numRotations;

			// Resize the inner vector and initialize each element
			result[i] = std::vector<std::shared_ptr<PhantomPlaintext>>(size);

			for (size_t j = 0; j < size; j++)
			{
				result[i][j] = std::make_shared<PhantomPlaintext>(); // Allocate a valid object
			}
		}

		// make sure the plaintext is created only with the necessary amount of moduli

		uint32_t towersToDrop = 0;
		uint32_t chain_idx = 1;
		if (L != 0)
		{
			towersToDrop = parms.coeff_modulus().size() - L - levelBudget;
		}

		for (uint32_t i = 0; i < towersToDrop; i++)
		{
			chain_idx++;
		}

		uint32_t level0 = towersToDrop;

		// we need to pre-compute the plaintexts in the extended basis P*Q
		std::vector<uint32_t> chainVector(levelBudget - flagRem + 1);

		for (int32_t s = 0; s < levelBudget - flagRem + 1; s++)
		{
			chainVector[s] = chain_idx;
			chain_idx++;
		}

		if (slots == M / 4)
		{
			// fully-packed
			auto coeff = CoeffDecodingCollapse(A, rotGroup, levelBudget, flag_i);

			for (int32_t s = 0; s < levelBudget - flagRem; s++)
			{
				for (int32_t i = 0; i < b; i++)
				{
					for (int32_t j = 0; j < g; j++)
					{
						if (g * i + j != int32_t(numRotations))
						{
							uint32_t rot = ReduceRotation(-g * i * (1 << (s * layersCollapse)), slots);
							if ((flagRem == 0) && (s == levelBudget - flagRem - 1))
							{
								// do the scaling only at the last set of coefficients
								for (uint32_t k = 0; k < slots; k++)
								{
									coeff[s][g * i + j][k] *= scale;
								}
							}

							auto rotateTemp = Rotate(coeff[s][g * i + j], rot);
							encoder_.encode_ext(cc, rotateTemp, m_scalingFactorsReal[level0 + s], *result[s][g * i + j], chainVector[s]);
						}
					}
				}
			}

			if (flagRem)
			{
				int32_t s = levelBudget - flagRem;
				for (int32_t i = 0; i < bRem; i++)
				{
					for (int32_t j = 0; j < gRem; j++)
					{
						if (gRem * i + j != int32_t(numRotationsRem))
						{
							uint32_t rot = ReduceRotation(-gRem * i * (1 << (s * layersCollapse)), slots);
							for (uint32_t k = 0; k < slots; k++)
							{
								coeff[s][gRem * i + j][k] *= scale;
							}

							auto rotateTemp = Rotate(coeff[s][gRem * i + j], rot);
							encoder_.encode_ext(cc, rotateTemp, m_scalingFactorsReal[level0 + s], *result[s][gRem * i + j], chainVector[s]);
						}
					}
				}
			}
		}
		else
		{
			//------------------------------------------------------------------------------
			// sparsely-packed mode
			//------------------------------------------------------------------------------

			auto coeff = CoeffDecodingCollapse(A, rotGroup, levelBudget, false);
			auto coeffi = CoeffDecodingCollapse(A, rotGroup, levelBudget, true);

			for (int32_t s = 0; s < levelBudget - flagRem; s++)
			{
				for (int32_t i = 0; i < b; i++)
				{
					for (int32_t j = 0; j < g; j++)
					{
						if (g * i + j != int32_t(numRotations))
						{
							uint32_t rot = ReduceRotation(-g * i * (1 << (s * layersCollapse)), M / 4);
							// concatenate the coefficients horizontally on their third dimension, which corresponds to the # of slots
							auto clearTemp = coeff[s][g * i + j];
							auto clearTempi = coeffi[s][g * i + j];
							clearTemp.insert(clearTemp.end(), clearTempi.begin(), clearTempi.end());
							if ((flagRem == 0) && (s == levelBudget - flagRem - 1))
							{
								// do the scaling only at the last set of coefficients
								for (uint32_t k = 0; k < clearTemp.size(); k++)
								{
									clearTemp[k] *= scale;
								}
							}

							auto rotateTemp = Rotate(clearTemp, rot);
							encoder_.encode_sparse_ext(cc, rotateTemp, m_scalingFactorsReal[level0 + s], *result[s][g * i + j], chainVector[s]);
						}
					}
				}
			}

			if (flagRem)
			{
				int32_t s = levelBudget - flagRem;
				for (int32_t i = 0; i < bRem; i++)
				{
					for (int32_t j = 0; j < gRem; j++)
					{
						if (gRem * i + j != int32_t(numRotationsRem))
						{
							uint32_t rot = ReduceRotation(-gRem * i * (1 << (s * layersCollapse)), M / 4);
							// concatenate the coefficients horizontally on their third dimension, which corresponds to the # of slots
							auto clearTemp = coeff[s][gRem * i + j];
							auto clearTempi = coeffi[s][gRem * i + j];
							clearTemp.insert(clearTemp.end(), clearTempi.begin(), clearTempi.end());
							for (uint32_t k = 0; k < clearTemp.size(); k++)
							{
								clearTemp[k] *= scale;
							}

							auto rotateTemp = Rotate(clearTemp, rot);
							encoder_.encode_sparse_ext(cc, rotateTemp, m_scalingFactorsReal[level0 + s], *result[s][gRem * i + j], chainVector[s]);
						}
					}
				}
			}
		}
		return result;
	}

	uint32_t FHECKKSRNS::GetBootstrapDepth(const std::vector<uint32_t> &levelBudget)
	{
		uint32_t approxModDepth = GetModDepthInternal();
		return approxModDepth + levelBudget[0] + levelBudget[1];
	}

	uint32_t FHECKKSRNS::GetModDepthInternal()
	{
		return GetMultiplicativeDepthByCoeffVector(g_coefficientsUniform, false) + R_UNIFORM;
	}

	//------------------------------------------------------------------------------
	// Find Rotation Indices
	//------------------------------------------------------------------------------

	std::vector<int32_t> FHECKKSRNS::FindBootstrapRotationIndices(uint32_t slots, uint32_t M)
	{
		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		std::vector<int32_t> fullIndexList;

		bool isLTBootstrap = (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) &&
							 (precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);

		if (isLTBootstrap)
		{
			// fullIndexList = FindLinearTransformRotationIndices(slots, M);
		}
		else
		{
			fullIndexList = FindCoeffsToSlotsRotationIndices(slots, M);

			std::vector<int32_t> indexListStC = FindSlotsToCoeffsRotationIndices(slots, M);
			fullIndexList.insert(fullIndexList.end(), indexListStC.begin(), indexListStC.end());
		}

		// Remove possible duplicates
		sort(fullIndexList.begin(), fullIndexList.end());
		fullIndexList.erase(unique(fullIndexList.begin(), fullIndexList.end()), fullIndexList.end());

		// remove automorphisms corresponding to 0
		fullIndexList.erase(std::remove(fullIndexList.begin(), fullIndexList.end(), 0), fullIndexList.end());
		fullIndexList.erase(std::remove(fullIndexList.begin(), fullIndexList.end(), M / 4), fullIndexList.end());

		return fullIndexList;
	}

	std::vector<int32_t> FHECKKSRNS::FindCoeffsToSlotsRotationIndices(uint32_t slots, uint32_t M)
	{
		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		std::vector<int32_t> indexList;

		int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t stop;
		int32_t flagRem;
		if (remCollapse == 0)
		{
			stop = -1;
			flagRem = 0;
		}
		else
		{
			stop = 0;
			flagRem = 1;
		}

		// Computing all indices for baby-step giant-step procedure for encoding and decoding
		indexList.reserve(b + g - 2 + bRem + gRem - 2 + 1 + M);

		for (int32_t s = int32_t(levelBudget) - 1; s > stop; s--)
		{
			for (int32_t j = 0; j < g; j++)
			{
				indexList.emplace_back(ReduceRotation(
					(j - int32_t((numRotations + 1) / 2) + 1) * (1 << ((s - flagRem) * layersCollapse + remCollapse)),
					slots));
			}

			for (int32_t i = 0; i < b; i++)
			{
				indexList.emplace_back(
					ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4));
			}
		}

		if (flagRem)
		{
			for (int32_t j = 0; j < gRem; j++)
			{
				indexList.emplace_back(ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots));
			}
			for (int32_t i = 0; i < bRem; i++)
			{
				indexList.emplace_back(ReduceRotation(gRem * i, M / 4));
			}
		}

		uint32_t m = slots * 4;
		// additional automorphisms are needed for sparse bootstrapping
		if (m != M)
		{
			for (uint32_t j = 1; j < M / m; j <<= 1)
			{
				indexList.emplace_back(j * slots);
			}
		}

		// Remove possible duplicates
		sort(indexList.begin(), indexList.end());
		indexList.erase(unique(indexList.begin(), indexList.end()), indexList.end());

		// remove automorphisms corresponding to 0
		indexList.erase(std::remove(indexList.begin(), indexList.end(), 0), indexList.end());
		indexList.erase(std::remove(indexList.begin(), indexList.end(), M / 4), indexList.end());

		return indexList;
	}

	std::vector<int32_t> FHECKKSRNS::FindSlotsToCoeffsRotationIndices(uint32_t slots, uint32_t M)
	{
		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		std::vector<int32_t> indexList;

		int32_t levelBudget = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t flagRem;
		if (remCollapse == 0)
		{
			flagRem = 0;
		}
		else
		{
			flagRem = 1;
		}

		// Computing all indices for baby-step giant-step procedure for encoding and decoding
		indexList.reserve(b + g - 2 + bRem + gRem - 2 + 1 + M);

		for (int32_t s = 0; s < int32_t(levelBudget) - flagRem; s++)
		{
			for (int32_t j = 0; j < g; j++)
			{
				indexList.emplace_back(
					ReduceRotation((j - (numRotations + 1) / 2 + 1) * (1 << (s * layersCollapse)), M / 4));
			}
			for (int32_t i = 0; i < b; i++)
			{
				indexList.emplace_back(ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4));
			}
		}

		if (flagRem)
		{
			int32_t s = int32_t(levelBudget) - flagRem;
			for (int32_t j = 0; j < gRem; j++)
			{
				indexList.emplace_back(
					ReduceRotation((j - (numRotationsRem + 1) / 2 + 1) * (1 << (s * layersCollapse)), M / 4));
			}
			for (int32_t i = 0; i < bRem; i++)
			{
				indexList.emplace_back(ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4));
			}
		}

		uint32_t m = slots * 4;
		// additional automorphisms are needed for sparse bootstrapping
		if (m != M)
		{
			for (uint32_t j = 1; j < M / m; j <<= 1)
			{
				indexList.emplace_back(j * slots);
			}
		}

		// Remove possible duplicates
		sort(indexList.begin(), indexList.end());
		indexList.erase(unique(indexList.begin(), indexList.end()), indexList.end());

		// remove automorphisms corresponding to 0
		indexList.erase(std::remove(indexList.begin(), indexList.end(), 0), indexList.end());
		indexList.erase(std::remove(indexList.begin(), indexList.end(), M / 4), indexList.end());

		return indexList;
	}

	void FHECKKSRNS::EvalBootstrapKeyGen(PhantomSecretKey &secret_key, PhantomContext &context, uint32_t numSlots)
	{
		auto &context_data = context.get_context_data(context.get_first_index());
		auto &key_context_data = context.get_context_data(0);
		auto &parms = context_data.parms();

		size_t poly_degree = parms.poly_modulus_degree();
		uint32_t M = 2 * poly_degree;

		auto vec = FindBootstrapRotationIndices(numSlots, M);

		galois_keys = secret_key.EvalRotateKeyGen(context, vec);
	}

	void FHECKKSRNS::EvalMultKeyGen(PhantomSecretKey &secret_key, PhantomContext &context)
	{
		mul_key = secret_key.gen_relinkey(context);
	}

	PhantomCiphertext FHECKKSRNS::EvalBootstrap(PhantomCiphertext &ciphertext, const PhantomContext &context, uint32_t numSlots,
												uint32_t numIterations, uint32_t precision)
	{

		auto &context_data = context.get_context_data(context.get_first_index());
		auto &key_context_data = context.get_context_data(0);
		auto &parms = context_data.parms();

		size_t poly_degree = parms.poly_modulus_degree();
		int N = poly_degree;
		int M = 2 * poly_degree;

		uint32_t L0 = parms.coeff_modulus().size();
		auto initSizeQ = context.get_context_data(ciphertext.chain_index()).parms().coeff_modulus().size();

		if (numIterations > 1)
		{
			// Step 1: Get the input.
			uint32_t powerOfTwoModulus = 1 << precision;

			// Step 2: Scale up by powerOfTwoModulus, and extend the modulus to powerOfTwoModulus * q.
			// Note that we extend the modulus implicitly without any code calls because the value always stays 0.
			PhantomCiphertext ctScaledUp = ciphertext;
			// We multiply by powerOfTwoModulus, and leave the last CRT value to be 0 (mod powerOfTwoModulus).
			MultByIntegerInPlace(context, ctScaledUp, powerOfTwoModulus);
			// RaisedCipher.set_chain_index(context.get_first_index());

			// Step 3: Bootstrap the initial ciphertext.
			PhantomCiphertext ctInitialBootstrap = EvalBootstrap(ciphertext, context, numSlots, numIterations - 1, precision);
			EvalModReduceInPlace(context, ctInitialBootstrap, 1);

			// Step 4: Scale up by powerOfTwoModulus.
			MultByIntegerInPlace(context, ctInitialBootstrap, powerOfTwoModulus);

			// Step 5: Mod-down to powerOfTwoModulus * q
			// We mod down, and leave the last CRT value to be 0 because it's divisible by powerOfTwoModulus.
			auto ctBootstrappedScaledDown = ctInitialBootstrap;
			auto bootstrappingSizeQ = context.get_context_data(ctBootstrappedScaledDown.chain_index()).parms().coeff_modulus().size();

			// If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
			if (bootstrappingSizeQ <= initSizeQ)
			{
				return ciphertext;
			}

			ModSwitchLevelInPlace(context, ctBootstrappedScaledDown, bootstrappingSizeQ - initSizeQ);

			// Step 6 and 7: Calculate the bootstrapping error by subtracting the original ciphertext from the bootstrapped ciphertext. Mod down to q is done implicitly.
			auto ctBootstrappingError = ctBootstrappedScaledDown;
			EvalSubAutoInplace(context, ctBootstrappingError, ctScaledUp, m_scalingFactorsReal, m_scalingFactorsRealBig);
			// Step 8: Bootstrap the error.

			auto ctBootstrappedError = EvalBootstrap(ctBootstrappingError, context, numSlots, 1, 0);
			EvalModReduceInPlace(context, ctBootstrappedError, 1);

			// Step 9: Subtract the bootstrapped error from the initial bootstrap to get even lower error.
			auto finalCiphertext = ctInitialBootstrap;
			EvalSubAutoInplace(context, finalCiphertext, ctBootstrappedError, m_scalingFactorsReal, m_scalingFactorsRealBig);

			// Step 10: Scale back down by powerOfTwoModulus to get the original message.
			EvalMultConstInplace(context, finalCiphertext, static_cast<double>(1) / powerOfTwoModulus, m_scalingFactorsReal);

			return finalCiphertext;
		}

		uint32_t slots = (numSlots == 0) ? poly_degree / 2 : numSlots;

		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup and then EvalBootstrapKeyGen to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		double qDouble = static_cast<double>(parms.coeff_modulus()[0].value());

		double powP = std::pow(2, 59);

		int32_t deg = std::round(std::log2(qDouble / powP));
		if (deg > static_cast<int32_t>(m_correctionFactor))
		{
			throwError("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
					   std::to_string(m_correctionFactor) + "].");
		}

		uint32_t correction = m_correctionFactor - deg;
		double post = std::pow(2, static_cast<double>(deg));

		double pre = 1. / post;
		uint64_t scalar = std::llround(post);

		//------------------------------------------------------------------------------
		// RAISING THE MODULUS
		//------------------------------------------------------------------------------

		// In FLEXIBLEAUTO, raising the ciphertext to a larger number
		// of towers is a bit more complex, because we need to adjust
		// it's scaling factor to the one that corresponds to the level
		// it's being raised to.
		// Increasing the modulus

		PhantomCiphertext raised = ciphertext;
		EvalModReduceInPlace(context, raised, raised.GetNoiseScaleDeg() - 1);
		AdjustCiphertext(raised, context, correction);

		raised = RaiseMod(context, raised);

		//------------------------------------------------------------------------------
		// SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
		//------------------------------------------------------------------------------

		// Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)

		std::vector<double> coefficients = g_coefficientsUniform;
		double k = K_UNIFORM;

		double constantEvalMult = pre * (1.0 / (k * N));

		EvalMultConstInplace(context, raised, constantEvalMult, m_scalingFactorsReal);

		// no linear transformations are needed for Chebyshev series as the range has been normalized to [-1,1]
		double coeffLowerBound = -1;
		double coeffUpperBound = 1;

		PhantomCiphertext ctxtDec;

		bool isLTBootstrap = (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) &&
							 (precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);
		if (slots == M / 4)
		{
			//------------------------------------------------------------------------------
			// FULLY PACKED CASE
			//------------------------------------------------------------------------------
			//------------------------------------------------------------------------------
			// Running CoeffToSlot
			//------------------------------------------------------------------------------

			// need to call internal modular reduction so it also works for FLEXIBLEAUTO
			EvalModReduceInPlace(context, raised, 1);

			// only one linear transform is needed as the other one can be derived
			// auto ctxtEnc = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0hatTPre, raised, context) :
			//                                 EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised, context);

			auto ctxtEnc = EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised, context);
			PhantomCiphertext conj;
			EvalConjFused(context, galois_keys, ctxtEnc, conj);

			auto ctxtEncI = EvalSubAuto(context, ctxtEnc, conj, m_scalingFactorsReal, m_scalingFactorsRealBig);

			EvalAddAutoInplace(context, ctxtEnc, conj, m_scalingFactorsReal, m_scalingFactorsRealBig);
			MultByMonomialInPlace(context, ctxtEncI, 3 * M / 4);

			if (ctxtEnc.GetNoiseScaleDeg() == 2)
			{
				EvalModReduceInPlace(context, ctxtEnc, 1);
				EvalModReduceInPlace(context, ctxtEncI, 1);
			}

			//------------------------------------------------------------------------------
			// Running Approximate Mod Reduction
			//------------------------------------------------------------------------------

			// Evaluate Chebyshev series for the sine wave
			ctxtEnc = EvalChebyshevSeries(context, mul_key, ctxtEnc, coefficients, coeffLowerBound,
										  coeffUpperBound, m_scalingFactorsReal, m_scalingFactorsRealBig);
			ctxtEncI = EvalChebyshevSeries(context, mul_key, ctxtEncI, coefficients, coeffLowerBound,
										   coeffUpperBound, m_scalingFactorsReal, m_scalingFactorsRealBig);

			// Double-angle iterations
			EvalModReduceInPlace(context, ctxtEnc, 1);
			EvalModReduceInPlace(context, ctxtEncI, 1);

			uint32_t numIter = R_UNIFORM;

			ApplyDoubleAngleIterations(context, ctxtEnc, numIter);
			ApplyDoubleAngleIterations(context, ctxtEncI, numIter);

			MultByMonomialInPlace(context, ctxtEncI, M / 4);
			EvalAddAutoInplace(context, ctxtEnc, ctxtEncI, m_scalingFactorsReal, m_scalingFactorsRealBig);

			// scale the message back up after Chebyshev interpolation
			MultByIntegerInPlace(context, ctxtEnc, scalar);
			//------------------------------------------------------------------------------
			// Running SlotToCoeff
			//------------------------------------------------------------------------------

			// In the case of FLEXIBLEAUTO, we need one extra tower
			// TODO: See if we can remove the extra level in FLEXIBLEAUTO

			EvalModReduceInPlace(context, ctxtEnc, 1);

			// Only one linear transform is needed
			// ctxtDec = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0Pre, ctxtEnc, context) :
			//                             EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc, context);
			ctxtDec = EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc, context);
		}
		else
		{
			//------------------------------------------------------------------------------
			// SPARSELY PACKED CASE
			//------------------------------------------------------------------------------

			//------------------------------------------------------------------------------
			// Running PartialSum
			//------------------------------------------------------------------------------

			for (uint32_t j = 1; j < N / (2 * slots); j <<= 1)
			{
				PhantomCiphertext temp;
				EvalRotateFused(context, galois_keys, raised, temp, j * slots);
				EvalAddAutoInplace(context, raised, temp, m_scalingFactorsReal, m_scalingFactorsRealBig);
			}

			//------------------------------------------------------------------------------
			// Running CoeffsToSlots
			//------------------------------------------------------------------------------

			EvalModReduceInPlace(context, raised, 1);

			// auto ctxtEnc = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0hatTPre, raised, context) :
			//                                 EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised, context);
			auto ctxtEnc = EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised, context, slots);
			PhantomCiphertext conj;
			EvalConjFused(context, galois_keys, ctxtEnc, conj);
			EvalAddAutoInplace(context, ctxtEnc, conj, m_scalingFactorsReal, m_scalingFactorsRealBig);

			if (ctxtEnc.GetNoiseScaleDeg() == 2)
			{
				EvalModReduceInPlace(context, ctxtEnc, 1);
			}
			//------------------------------------------------------------------------------
			// Running Approximate Mod Reduction
			//------------------------------------------------------------------------------

			// Evaluate Chebyshev series for the sine wave
			ctxtEnc = EvalChebyshevSeries(context, mul_key, ctxtEnc, coefficients,
										  coeffLowerBound, coeffUpperBound, m_scalingFactorsReal, m_scalingFactorsRealBig);

			// Double-angle iterations
			EvalModReduceInPlace(context, ctxtEnc, 1);
			uint32_t numIter = R_UNIFORM;
			ApplyDoubleAngleIterations(context, ctxtEnc, numIter);

			// scale the message back up after Chebyshev interpolation
			MultByIntegerInPlace(context, ctxtEnc, scalar);

			//------------------------------------------------------------------------------
			// Running SlotsToCoeffs
			//------------------------------------------------------------------------------

			// In the case of FLEXIBLEAUTO, we need one extra tower
			// TODO: See if we can remove the extra level in FLEXIBLEAUTO
			EvalModReduceInPlace(context, ctxtEnc, 1);

			// linear transform for decoding
			// ctxtDec = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0Pre, ctxtEnc, context) :
			//                             EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc, context);

			ctxtDec = EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc, context, slots);

			PhantomCiphertext rotated;
			EvalRotateFused(context, galois_keys, ctxtDec, rotated, slots);
			EvalAddAutoInplace(context, ctxtDec, rotated, m_scalingFactorsReal, m_scalingFactorsRealBig);
		}

		// 64-bit only: scale back the message to its original scale.
		uint64_t corFactor = (uint64_t)1 << std::llround(correction);
		MultByIntegerInPlace(context, ctxtDec, corFactor);
		size_t bootstrappingNumTowers = context.get_context_data(ctxtDec.chain_index()).gpu_rns_tool().base_Ql().size();

		// If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
		if (bootstrappingNumTowers <= initSizeQ)
		{
			// std::cout << "Bootstrapping failed: the number of towers after bootstrapping is less than the original ciphertext." << std::endl;
			// std::cout << "Original ciphertext towers: " << initSizeQ << std::endl;
			// std::cout << "Bootstrapped ciphertext towers: " << bootstrappingNumTowers << std::endl;
			// std::cout << "Returning the original ciphertext." << std::endl;
			PhantomCiphertext New = ciphertext;
			return New;
		}

		return ctxtDec;
	}

	void FHECKKSRNS::AdjustCiphertext(PhantomCiphertext &ciphertext, const PhantomContext &context, double correction) const
	{

		auto &context_data = context.get_context_data(ciphertext.chain_index());
		auto &parms = context_data.parms();

		uint32_t lvl = 0;
		double targetSF = m_scalingFactorsReal[lvl];
		double sourceSF = ciphertext.scale();
		uint32_t numTowers = parms.coeff_modulus().size();
		double modToDrop = static_cast<double>(context.get_context_data(0).parms().coeff_modulus()[numTowers - 1].value());

		// in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
		// a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
		// So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
		// for NATIVEINT = 128.

		// Scaling down the message by a correction factor to emulate using a larger q0.
		// This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
		double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * std::pow(2, -correction);
		EvalMultConstInplace(context, ciphertext, adjustmentFactor, m_scalingFactorsReal);

		EvalModReduceInPlace(context, ciphertext, 1);
		ciphertext.set_scale(targetSF);
	}

	PhantomCiphertext FHECKKSRNS::EvalCoeffsToSlots(const std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> &A,
													const PhantomCiphertext &ctxt, const PhantomContext &context, uint32_t numSlots) const
	{

		auto &context_data = context.get_context_data(context.get_first_index());
		auto &key_context_data = context.get_context_data(0);
		auto &parms = context_data.parms();

		size_t poly_degree = parms.poly_modulus_degree();
		uint32_t N = poly_degree;
		uint32_t M = 2 * poly_degree;

		uint32_t slots = (numSlots == 0) ? N / 2 : numSlots;

		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup and EvalBootstrapKeyGen to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t stop = -1;
		int32_t flagRem = 0;

		if (remCollapse != 0)
		{
			stop = 0;
			flagRem = 1;
		}

		size_t size_Ql = context.get_context_data(ctxt.chain_index()).gpu_rns_tool().base_Ql().size();

		size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
		size_t size_P = context.get_context_data(0).parms().special_modulus_size();
		auto modulus_QP = context.gpu_rns_tables().modulus();
		// precompute the inner and outer rotations
		std::vector<std::vector<int32_t>> rot_in(levelBudget);
		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			if (flagRem == 1 && i == 0)
			{
				// remainder corresponds to index 0 in encoding and to last index in decoding
				rot_in[i] = std::vector<int32_t>(numRotationsRem + 1);
			}
			else
			{
				rot_in[i] = std::vector<int32_t>(numRotations + 1);
			}
		}

		std::vector<std::vector<int32_t>> rot_out(levelBudget);
		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			rot_out[i] = std::vector<int32_t>(b + bRem);
		}

		for (int32_t si = levelBudget - 1; si > stop; si--)
		{
			for (int32_t j = 0; j < g; j++)
			{
				rot_in[si][j] = ReduceRotation(
					(j - int32_t((numRotations + 1) / 2) + 1) * (1 << ((si - flagRem) * layersCollapse + remCollapse)),
					slots);
			}

			for (int32_t i = 0; i < b; i++)
			{
				rot_out[si][i] = ReduceRotation((g * i) * (1 << ((si - flagRem) * layersCollapse + remCollapse)), M / 4);
			}
		}

		if (flagRem)
		{
			for (int32_t j = 0; j < gRem; j++)
			{
				rot_in[stop][j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
			}

			for (int32_t i = 0; i < bRem; i++)
			{
				rot_out[stop][i] = ReduceRotation((gRem * i), M / 4);
			}
		}

		PhantomCiphertext result = ctxt;
		// hoisted automorphisms
		for (int32_t si = levelBudget - 1; si > stop; si--)
		{
			if (si != levelBudget - 1)
			{
				EvalModReduceInPlace(context, result, 1);
			}

			// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
			auto digits = EvalFastRotationPrecompute(context, result);

			std::vector<PhantomCiphertext> fastRotation(g);
			for (int32_t j = 0; j < g; j++)
			{
				if (rot_in[si][j] != 0)
				{
					fastRotation[j] = EvalFastRotationExt(context, result, galois_keys, rot_in[si][j], digits, true);
				}
				else
				{
					fastRotation[j] = KeySwitchExt(context, result);
				}
			}

			PhantomCiphertext outer;
			PhantomCiphertext first;
			for (int32_t i = 0; i < b; i++)
			{
				// for the first iteration with j=0:
				int32_t G = g * i;

				PhantomCiphertext inner = EvalMultExt(context, fastRotation[0], *A[si][G]);

				// continue the loop
				for (int32_t j = 1; j < g; j++)
				{
					if ((G + j) != int32_t(numRotations))
					{
						EvalAddExtInPlace(context, inner, EvalMultExt(context, fastRotation[j], *A[si][G + j]));
					}
				}

				if (i == 0)
				{
					first = KeySwitchDownFirstElement(context, inner);
					outer = inner.clone();
					inner.share_data_with(outer);
					reset_poly_ext(context, outer.data(), outer.chain_index());
				}

				else
				{
					if (rot_out[si][i] != 0)
					{
						inner = KeySwitchDown(context, inner);
						// Find the automorphism index that corresponds to rotation index index.
						auto inner_rot = rotate_c0(context, inner, rot_out[si][i]);
						add_two_poly_inplace(context, first.data(), inner_rot.get(), inner.chain_index());

						auto innerDigits = EvalFastRotationPrecompute(context, inner);
						EvalAddExtInPlace(context, outer, EvalFastRotationExt(context, inner, galois_keys, rot_out[si][i], innerDigits, false));
					}
					else
					{

						add_two_poly_inplace(context, first.data(), KeySwitchDownFirstElement(context, inner).data(), inner.chain_index());

						uint64_t rns_coeff_count = outer.poly_modulus_degree() * outer.coeff_modulus_size();

						// Only Add c1
						add_two_poly_inplace_ext(context, outer.data() + rns_coeff_count, inner.data() + rns_coeff_count, inner.chain_index());
					}
				}
			}

			result = KeySwitchDown(context, outer);
			add_two_poly_inplace(context, result.data(), first.data(), result.chain_index());
		}

		if (flagRem)
		{
			EvalModReduceInPlace(context, result, 1);

			// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
			auto digits = EvalFastRotationPrecompute(context, result);
			std::vector<PhantomCiphertext> fastRotation(gRem);

			for (int32_t j = 0; j < gRem; j++)
			{
				if (rot_in[stop][j] != 0)
				{
					fastRotation[j] = EvalFastRotationExt(context, result, GetGaloisKey(), rot_in[stop][j], digits, true);
				}
				else
				{
					fastRotation[j] = KeySwitchExt(context, result);
				}
			}

			PhantomCiphertext outer;
			PhantomCiphertext first;
			for (int32_t i = 0; i < bRem; i++)
			{
				// for the first iteration with j=0:
				int32_t GRem = gRem * i;

				PhantomCiphertext inner = EvalMultExt(context, fastRotation[0], *A[stop][GRem]);

				// continue the loop
				for (int32_t j = 1; j < gRem; j++)
				{
					if ((GRem + j) != int32_t(numRotationsRem))
					{
						EvalAddExtInPlace(context, inner, EvalMultExt(context, fastRotation[j], *A[stop][GRem + j]));
					}
				}

				if (i == 0)
				{
					first = KeySwitchDownFirstElement(context, inner);
					outer = inner.clone();
					inner.share_data_with(outer);
					reset_poly_ext(context, outer.data(), outer.chain_index());
				}
				else
				{
					if (rot_out[stop][i] != 0)
					{
						inner = KeySwitchDown(context, inner);
						// Find the automorphism index that corresponds to rotation index index.
						auto inner_rot = rotate_c0(context, inner, rot_out[stop][i]);
						add_two_poly_inplace(context, first.data(), inner_rot.get(), inner.chain_index());

						auto innerDigits = EvalFastRotationPrecompute(context, inner);
						EvalAddExtInPlace(context, outer, EvalFastRotationExt(context, inner, galois_keys, rot_out[stop][i], innerDigits, false));
					}
					else
					{
						add_two_poly_inplace(context, first.data(), KeySwitchDownFirstElement(context, inner).data(), inner.chain_index());
						uint64_t rns_coeff_count = outer.poly_modulus_degree() * outer.coeff_modulus_size();
						add_two_poly_inplace_ext(context, outer.data() + rns_coeff_count, inner.data() + rns_coeff_count, inner.chain_index());
					}
				}
			}

			result = KeySwitchDown(context, outer);
			add_two_poly_inplace(context, result.data(), first.data(), result.chain_index());
		}

		return result;
	}

	PhantomCiphertext FHECKKSRNS::EvalSlotsToCoeffs(const std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> &A,
													const PhantomCiphertext &ctxt, const PhantomContext &context, uint32_t numSlots) const
	{

		auto &context_data = context.get_context_data(context.get_first_index());
		auto &key_context_data = context.get_context_data(0);
		auto &parms = context_data.parms();

		size_t poly_degree = parms.poly_modulus_degree();
		uint32_t N = poly_degree;
		uint32_t M = 2 * poly_degree;

		uint32_t slots = (numSlots == 0) ? N / 2 : numSlots;

		auto pair = m_bootPrecomMap.find(slots);
		if (pair == m_bootPrecomMap.end())
		{
			std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
								 std::string(" slots were not generated") +
								 std::string(" Need to call EvalBootstrapSetup and EvalBootstrapKeyGen to proceed"));
			throwError(errorMsg);
		}
		const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

		int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
		int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
		int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
		int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
		int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
		int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
		int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
		int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
		int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

		int32_t flagRem = 0;

		if (remCollapse != 0)
		{
			flagRem = 1;
		}

		size_t size_Ql = context.get_context_data(ctxt.chain_index()).gpu_rns_tool().base_Ql().size();
		size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
		size_t size_P = context.get_context_data(0).parms().special_modulus_size();
		auto modulus_QP = context.gpu_rns_tables().modulus();
		// precompute the inner and outer rotations
		std::vector<std::vector<int32_t>> rot_in(levelBudget);
		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			if (flagRem == 1 && i == uint32_t(levelBudget - 1))
			{
				// remainder corresponds to index 0 in encoding and to last index in decoding
				rot_in[i] = std::vector<int32_t>(numRotationsRem + 1);
			}
			else
			{
				rot_in[i] = std::vector<int32_t>(numRotations + 1);
			}
		}

		std::vector<std::vector<int32_t>> rot_out(levelBudget);
		for (uint32_t i = 0; i < uint32_t(levelBudget); i++)
		{
			rot_out[i] = std::vector<int32_t>(b + bRem);
		}

		for (int32_t si = 0; si < levelBudget - flagRem; si++)
		{
			for (int32_t j = 0; j < g; j++)
			{
				rot_in[si][j] =
					ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) * (1 << (si * layersCollapse)), M / 4);
			}

			for (int32_t i = 0; i < b; i++)
			{
				rot_out[si][i] = ReduceRotation((g * i) * (1 << (si * layersCollapse)), M / 4);
			}
		}
		// return ctxt; //error here

		if (flagRem)
		{
			int32_t si = levelBudget - flagRem;
			for (int32_t j = 0; j < gRem; j++)
			{
				rot_in[si][j] =
					ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (si * layersCollapse)), M / 4);
			}

			for (int32_t i = 0; i < bRem; i++)
			{
				rot_out[si][i] = ReduceRotation((gRem * i) * (1 << (si * layersCollapse)), M / 4);
			}
		}
		PhantomCiphertext result = ctxt;
		// hoisted automorphisms
		for (int32_t si = 0; si < levelBudget - flagRem; si++)
		{
			if (si != 0)
			{
				EvalModReduceInPlace(context, result, 1);
			}

			// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
			auto digits = EvalFastRotationPrecompute(context, result);

			std::vector<PhantomCiphertext> fastRotation(g);

			for (int32_t j = 0; j < g; j++)
			{
				if (rot_in[si][j] != 0)
				{
					fastRotation[j] = EvalFastRotationExt(context, result, galois_keys, rot_in[si][j], digits, true);
				}
				else
				{
					fastRotation[j] = KeySwitchExt(context, result);
				}
			}

			PhantomCiphertext outer;
			PhantomCiphertext first;
			for (int32_t i = 0; i < b; i++)
			{
				// for the first iteration with j=0:
				int32_t G = g * i;
				PhantomCiphertext inner = EvalMultExt(context, fastRotation[0], *A[si][G]);
				// continue the loop
				for (int32_t j = 1; j < g; j++)
				{
					if ((G + j) != int32_t(numRotations))
					{
						EvalAddExtInPlace(context, inner, EvalMultExt(context, fastRotation[j], *A[si][G + j]));
					}
				}

				if (i == 0)
				{
					first = KeySwitchDownFirstElement(context, inner);
					outer = inner.clone();
					inner.share_data_with(outer);
					reset_poly_ext(context, outer.data(), outer.chain_index());
				}
				else
				{
					if (rot_out[si][i] != 0)
					{
						inner = KeySwitchDown(context, inner);
						// Find the automorphism index that corresponds to rotation index index.
						auto inner_rot = rotate_c0(context, inner, rot_out[si][i]);
						add_two_poly_inplace(context, first.data(), inner_rot.get(), inner.chain_index());
						auto innerDigits = EvalFastRotationPrecompute(context, inner);

						EvalAddExtInPlace(context, outer, EvalFastRotationExt(context, inner, galois_keys, rot_out[si][i], innerDigits, false));
					}
					else
					{
						add_two_poly_inplace(context, first.data(), KeySwitchDownFirstElement(context, inner).data(), inner.chain_index());
						uint64_t rns_coeff_count = outer.poly_modulus_degree() * outer.coeff_modulus_size();
						add_two_poly_inplace_ext(context, outer.data() + rns_coeff_count, inner.data() + rns_coeff_count, inner.chain_index());
					}
				}
			}

			result = KeySwitchDown(context, outer);

			add_two_poly_inplace(context, result.data(), first.data(), result.chain_index());
		}

		if (flagRem)
		{
			EvalModReduceInPlace(context, result, 1);

			// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
			auto digits = EvalFastRotationPrecompute(context, result);
			std::vector<PhantomCiphertext> fastRotation(gRem);
			int32_t si = levelBudget - flagRem;

			for (int32_t j = 0; j < gRem; j++)
			{
				if (rot_in[si][j] != 0)
				{
					fastRotation[j] = EvalFastRotationExt(context, result, GetGaloisKey(), rot_in[si][j], digits, true);
				}
				else
				{
					fastRotation[j] = KeySwitchExt(context, result);
				}
			}

			PhantomCiphertext outer;
			PhantomCiphertext first;
			for (int32_t i = 0; i < bRem; i++)
			{
				// for the first iteration with j=0:
				int32_t GRem = gRem * i;
				PhantomCiphertext inner = EvalMultExt(context, fastRotation[0], *A[si][GRem]);
				// continue the loop
				for (int32_t j = 1; j < gRem; j++)
				{
					if ((GRem + j) != int32_t(numRotationsRem))
					{
						EvalAddExtInPlace(context, inner, EvalMultExt(context, fastRotation[j], *A[si][GRem + j]));
					}
				}

				if (i == 0)
				{

					first = KeySwitchDownFirstElement(context, inner);
					outer = inner.clone();
					inner.share_data_with(outer);
					reset_poly_ext(context, outer.data(), outer.chain_index());
				}
				else
				{

					if (rot_out[si][i] != 0)
					{

						inner = KeySwitchDown(context, inner);

						// Find the automorphism index that corresponds to rotation index index.
						auto inner_rot = rotate_c0(context, inner, rot_out[si][i]);

						add_two_poly_inplace(context, first.data(), inner_rot.get(), inner.chain_index());

						auto innerDigits = EvalFastRotationPrecompute(context, inner);

						EvalAddExtInPlace(context, outer, EvalFastRotationExt(context, inner, galois_keys, rot_out[si][i], innerDigits, false));
					}
					else
					{

						add_two_poly_inplace(context, first.data(), KeySwitchDownFirstElement(context, inner).data(), inner.chain_index());
						uint64_t rns_coeff_count = outer.poly_modulus_degree() * outer.coeff_modulus_size();
						add_two_poly_inplace_ext(context, outer.data() + rns_coeff_count, inner.data() + rns_coeff_count, inner.chain_index());
					}
				}
			}

			result = KeySwitchDown(context, outer);

			add_two_poly_inplace(context, result.data(), first.data(), result.chain_index());
		}

		return result;
	}

	void FHECKKSRNS::ApplyDoubleAngleIterations(const PhantomContext &context, PhantomCiphertext &ciphertext, uint32_t numIt) const
	{

		int32_t r = numIt;
		for (int32_t j = 1; j < r + 1; j++)
		{
			EvalSquareInPlace(context, ciphertext, mul_key, m_scalingFactorsReal, m_scalingFactorsRealBig);
			ciphertext = EvalAddAuto(context, ciphertext, ciphertext, m_scalingFactorsReal, m_scalingFactorsRealBig);
			double scalar = -1.0 / std::pow((2.0 * M_PI), std::pow(2.0, j - r));
			EvalAddConstInPlaceWrap(context, ciphertext, scalar, m_scalingFactorsReal, m_scalingFactorsRealBig);
		}
	}
}