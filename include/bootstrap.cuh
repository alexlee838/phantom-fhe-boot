#pragma once

#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "cuda_wrapper.cuh"
#include "util.cuh"
#include "ckks.h"
#include "error_handle.cuh"

# define M_PI		3.14159265358979323846	/* pi */

namespace phantom {

	class CKKSBootstrapPrecom {
	public:
		CKKSBootstrapPrecom() {}

		CKKSBootstrapPrecom(const CKKSBootstrapPrecom& rhs) {
			m_dim1         = rhs.m_dim1;
			m_slots        = rhs.m_slots;
			m_paramsEnc    = rhs.m_paramsEnc;
			m_paramsDec    = rhs.m_paramsDec;
			m_U0Pre        = rhs.m_U0Pre;
			m_U0hatTPre    = rhs.m_U0hatTPre;
			m_U0PreFFT     = rhs.m_U0PreFFT;
			m_U0hatTPreFFT = rhs.m_U0hatTPreFFT;
		}

		CKKSBootstrapPrecom(CKKSBootstrapPrecom&& rhs) {
			m_dim1         = rhs.m_dim1;
			m_slots        = rhs.m_slots;
			m_paramsEnc    = std::move(rhs.m_paramsEnc);
			m_paramsDec    = std::move(rhs.m_paramsDec);
			m_U0Pre        = std::move(rhs.m_U0Pre);
			m_U0hatTPre    = std::move(rhs.m_U0hatTPre);
			m_U0PreFFT     = std::move(rhs.m_U0PreFFT);
			m_U0hatTPreFFT = std::move(rhs.m_U0hatTPreFFT);
		}

		virtual ~CKKSBootstrapPrecom() {}
		// the inner dimension in the baby-step giant-step strategy
		uint32_t m_dim1 = 0;

		// number of slots for which the bootstrapping is performed
		uint32_t m_slots = 0;

		// level budget for homomorphic encoding, number of layers to collapse in one level,
		// number of layers remaining to be collapsed in one level to have exactly the number
		// of levels specified in the level budget, the number of rotations in one level,
		// the baby step and giant step in the baby-step giant-step strategy, the number of
		// rotations in the remaining level, the baby step and giant step in the baby-step
		// giant-step strategy for the remaining level
		std::vector<int32_t> m_paramsEnc = std::vector<int32_t>(CKKS_BOOT_PARAMS::TOTAL_ELEMENTS, 0);

		// level budget for homomorphic decoding, number of layers to collapse in one level,
		// number of layers remaining to be collapsed in one level to have exactly the number
		// of levels specified in the level budget, the number of rotations in one level,
		// the baby step and giant step in the baby-step giant-step strategy, the number of
		// rotations in the remaining level, the baby step and giant step in the baby-step
		// giant-step strategy for the remaining level
		std::vector<int32_t> m_paramsDec = std::vector<int32_t>(CKKS_BOOT_PARAMS::TOTAL_ELEMENTS, 0);

		// Linear map U0; used in decoding
		std::vector<std::shared_ptr<PhantomPlaintext>> m_U0Pre;

		// Conj(U0^T); used in encoding
		std::vector<std::shared_ptr<PhantomPlaintext>> m_U0hatTPre;

		// coefficients corresponding to U0; used in decoding
		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> m_U0PreFFT;

		// coefficients corresponding to conj(U0^T); used in encoding
		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> m_U0hatTPreFFT;

	};

	class FHECKKSRNS {

	public:
		~FHECKKSRNS() {
			m_scalingFactorsReal.clear();
			m_scalingFactorsRealBig.clear();
			m_bootPrecomMap.clear();
		}
		FHECKKSRNS(PhantomCKKSEncoder  &encoder) :encoder_(encoder) {}

		//------------------------------------------------------------------------------
		// Bootstrap Wrapper
		//------------------------------------------------------------------------------

		void EvalBootstrapSetup(const PhantomContext& cc, std::vector<uint32_t>& levelBudget, const double& scale, 
								const std::vector<double>& m_scalingFactorsReal, const std::vector<double>& m_scalingFactorsRealBig,
								std::vector<uint32_t> dim1 = {0, 0}, uint32_t slots = 0, uint32_t correctionFactor = 0,
								bool precompute = true);

		void EvalBootstrapKeyGen(PhantomSecretKey& secret_key, PhantomContext& context, uint32_t numSlots);

		void EvalMultKeyGen(PhantomSecretKey& secret_key, PhantomContext& context);

		PhantomCiphertext EvalBootstrap(PhantomCiphertext& ciphertext, const PhantomContext &context, uint32_t numSlots = 0, uint32_t numIterations = 1,
										uint32_t precision = 0);

		

		//------------------------------------------------------------------------------
		// Find Rotation Indices
		//------------------------------------------------------------------------------

		std::vector<int32_t> FindBootstrapRotationIndices(uint32_t slots, uint32_t M);

		std::vector<int32_t> FindLinearTransformRotationIndices(uint32_t slots, uint32_t M);

		std::vector<int32_t> FindCoeffsToSlotsRotationIndices(uint32_t slots, uint32_t M);

		std::vector<int32_t> FindSlotsToCoeffsRotationIndices(uint32_t slots, uint32_t M);

		//------------------------------------------------------------------------------
		// Precomputations for CoeffsToSlots and SlotsToCoeffs
		//------------------------------------------------------------------------------

		std::vector<std::shared_ptr<PhantomPlaintext>> EvalLinearTransformPrecompute(const PhantomContext& cc,
																const std::vector<std::vector<std::complex<double>>>& A,
																double scale = 1, uint32_t L = 0) const;

		std::vector<std::shared_ptr<PhantomPlaintext>> EvalLinearTransformPrecompute(const PhantomContext& cc,
																const std::vector<std::vector<std::complex<double>>>& A,
																const std::vector<std::vector<std::complex<double>>>& B,
																uint32_t orientation = 0, double scale = 1,
																uint32_t L = 0);

		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> EvalCoeffsToSlotsPrecompute(const PhantomContext& cc,
																			const std::vector<std::complex<double>>& A,
																			const std::vector<uint32_t>& rotGroup, const std::vector<double>& m_scalingFactorsReal,
																			bool flag_i, double scale = 1,
																			uint32_t L = 0) const;

		std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>> EvalSlotsToCoeffsPrecompute(const PhantomContext& cc,
																			const std::vector<std::complex<double>>& A,
																			const std::vector<uint32_t>& rotGroup, const std::vector<double>& m_scalingFactorsReal,
																			bool flag_i, double scale = 1,
																			uint32_t L = 0) const;

		//------------------------------------------------------------------------------
		// EVALUATION: CoeffsToSlots and SlotsToCoeffs
		//------------------------------------------------------------------------------

		// std::shared_ptr<PhantomCiphertext> EvalLinearTransform(const std::vector<std::shared_ptr<PhantomPlaintext>>& A, const PhantomCiphertext& ct) const;

		PhantomCiphertext EvalCoeffsToSlots(const std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>& A,
			const PhantomCiphertext& ctxt, const PhantomContext& context, uint32_t numSlots = 0) const;

		PhantomCiphertext EvalSlotsToCoeffs(const std::vector<std::vector<std::shared_ptr<PhantomPlaintext>>>& A,
			const PhantomCiphertext& ctxt, const PhantomContext& context, uint32_t numSlots = 0) const;

		static uint32_t GetBootstrapDepth(const std::vector<uint32_t>& levelBudget); //UNIFORM_TERNARY always


		//------------------------------------------------------------------------------
		// Getter and Setter for Keys for Bootstrapping
		//------------------------------------------------------------------------------

		inline const PhantomRelinKey& GetMultKey() const {
			return mul_key;  // Avoids reference count increment
		}

		inline const PhantomGaloisKeyFused& GetGaloisKey() const {
			return galois_keys;
		}
		

	private:
		//------------------------------------------------------------------------------
		// Auxiliary Bootstrap Functions
		//------------------------------------------------------------------------------
		uint32_t GetBootstrapDepthInternal(uint32_t approxModDepth, const std::vector<uint32_t>& levelBudget,
										const PhantomContext& cc);
		static uint32_t GetModDepthInternal(); //UNIFORM_TERNARY always

		void AdjustCiphertext(PhantomCiphertext& ciphertext, const PhantomContext& context, double correction) const;

		void ApplyDoubleAngleIterations(const PhantomContext &context, PhantomCiphertext& ciphertext, uint32_t numIt) const;

		std::shared_ptr<PhantomPlaintext> MakeAuxPlaintext(const EncryptionParameters& params,
								const std::vector<std::complex<double>>& value, size_t noiseScaleDeg, uint32_t level,
								uint32_t slots) const;


		// std::shared_ptr<PhantomGaloisKey> ConjugateKeyGen(const PhantomSecretKey& privateKey) const;

		// std::shared_ptr<PhantomCiphertext> Conjugate(const PhantomCiphertext& ciphertext,
		// 							const std::map<uint32_t, std::shared_ptr<PhantomGaloisKey>>& evalKeys) const;


		const uint32_t K_SPARSE  = 28;   // upper bound for the number of overflows in the sparse secret case
		const uint32_t K_UNIFORM = 512;  // upper bound for the number of overflows in the uniform secret case
		static const uint32_t R_UNIFORM =
			6;  // number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
		static const uint32_t R_SPARSE =
			3;  // number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
		uint32_t m_correctionFactor = 0;  // correction factor, which we scale the message by to improve precision

		// key tuple is dim1, levelBudgetEnc, levelBudgetDec
		std::map<uint32_t, std::shared_ptr<CKKSBootstrapPrecom>> m_bootPrecomMap;

		// Scale Precompuation
		std::vector<double> m_scalingFactorsReal;
		std::vector<double> m_scalingFactorsRealBig;

		// Chebyshev series coefficients for the SPARSE case
		static const inline std::vector<double> g_coefficientsSparse{
			-0.18646470117093214,   0.036680543700430925,    -0.20323558926782626,     0.029327390306199311,
			-0.24346234149506416,   0.011710240188138248,    -0.27023281815251715,     -0.017621188001030602,
			-0.21383614034992021,   -0.048567932060728937,   -0.013982336571484519,    -0.051097367628344978,
			0.24300487324019346,    0.0016547743046161035,   0.23316923792642233,      0.060707936480887646,
			-0.18317928363421143,   0.0076878773048247966,   -0.24293447776635235,     -0.071417413140564698,
			0.37747441314067182,    0.065154496937795681,    -0.24810721693607704,     -0.033588418808958603,
			0.10510660697380972,    0.012045222815124426,    -0.032574751830745423,    -0.0032761730196023873,
			0.0078689491066424744,  0.00070965574480802061,  -0.0015405394287521192,   -0.00012640521062948649,
			0.00025108496615830787, 0.000018944629154033562, -0.000034753284216308228, -2.4309868106111825e-6,
			4.1486274737866247e-6,  2.7079833113674568e-7,   -4.3245388569898879e-7,   -2.6482744214856919e-8,
			3.9770028771436554e-8,  2.2951153557906580e-9,   -3.2556026220554990e-9,   -1.7691071323926939e-10,
			2.5459052150406730e-10};

		// Chebyshev series coefficients for the OPTIMIZED/uniform case
		static const inline std::vector<double> g_coefficientsUniform{
			0.15421426400235561,    -0.0037671538417132409,  0.16032011744533031,      -0.0034539657223742453,
			0.17711481926851286,    -0.0027619720033372291,  0.19949802549604084,      -0.0015928034845171929,
			0.21756948616367638,    0.00010729951647566607,  0.21600427371240055,      0.0022171399198851363,
			0.17647500259573556,    0.0042856217194480991,   0.086174491919472254,     0.0054640252312780444,
			-0.046667988130649173,  0.0047346914623733714,   -0.17712686172280406,     0.0016205080004247200,
			-0.22703114241338604,   -0.0028145845916205865,  -0.13123089730288540,     -0.0056345646688793190,
			0.078818395388692147,   -0.0037868875028868542,  0.23226434602675575,      0.0021116338645426574,
			0.13985510526186795,    0.0059365649669377071,   -0.13918475289368595,     0.0018580676740836374,
			-0.23254376365752788,   -0.0054103844866927788,  0.056840618403875359,     -0.0035227192748552472,
			0.25667909012207590,    0.0055029673963982112,   -0.073334392714092062,    0.0027810273357488265,
			-0.24912792167850559,   -0.0069524866497120566,  0.21288810409948347,      0.0017810057298691725,
			0.088760951809475269,   0.0055957188940032095,   -0.31937177676259115,     -0.0087539416335935556,
			0.34748800245527145,    0.0075378299617709235,   -0.25116537379803394,     -0.0047285674679876204,
			0.13970502851683486,    0.0023672533925155220,   -0.063649401080083698,    -0.00098993213448982727,
			0.024597838934816905,   0.00035553235917057483,  -0.0082485030307578155,   -0.00011176184313622549,
			0.0024390574829093264,  0.000031180384864488629, -0.00064373524734389861,  -7.8036008952377965e-6,
			0.00015310015145922058, 1.7670804180220134e-6,   -0.000033066844379476900, -3.6460909134279425e-7,
			6.5276969021754105e-6,  6.8957843666189918e-8,   -1.1842811187642386e-6,   -1.2015133285307312e-8,
			1.9839339947648331e-7,  1.9372045971100854e-9,   -3.0815418032523593e-8,   -2.9013806338735810e-10,
			4.4540904298173700e-9,  4.0505136697916078e-11,  -6.0104912807134771e-10,  -5.2873323696828491e-12,
			7.5943206779351725e-11, 6.4679566322060472e-13,  -9.0081200925539902e-12,  -7.4396949275292252e-14,
			1.0057423059167244e-12, 8.1701187638005194e-15,  -1.0611736208855373e-13,  -8.9597492970451533e-16,
			1.1421575296031385e-14};

		//------------------------------------------------------------------------------
		// Keys for Bootstrapping
		//------------------------------------------------------------------------------

		PhantomRelinKey mul_key;
		PhantomGaloisKeyFused galois_keys;

		// Encoder for Sparse Bootstrapping
		PhantomCKKSEncoder &encoder_;


	};
}