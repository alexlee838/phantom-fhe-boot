#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <complex>
#include <memory> 
#include <cuComplex.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>


#include "error_handle.cuh"
#include "./host/numth.h"

// A function to return a seeded random number generator.
inline std::mt19937 &generator() {
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// A function to generate integers in the range [min, max]
inline int my_rand_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator());
}

inline void print_timer_banner() {
    // print header
    std::cout << "function,trials,median time (us),mean time (us),std. dev." << std::endl;
}

class ChronoTimer {
public:
    explicit ChronoTimer(std::string func_name) {
        func_name_ = std::move(func_name);
    }

    ~ChronoTimer() {
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto median_time = median(time_);
        auto min_time = min(time_);
        auto stddev = std_dev(time_);
        std::cout << func_name_ << ","
                  << n_trials << ","
                  << median_time << ","
                  << mean_time << std::endl;
    }

    inline void start() {
        start_point_ = std::chrono::steady_clock::now();
    }

    inline void stop() {
        stop_point_ = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::micro> elapsed_time = stop_point_ - start_point_;
        time_.emplace_back(elapsed_time.count());
    }

private:
    std::string func_name_;

    std::chrono::time_point<std::chrono::steady_clock> start_point_, stop_point_;
    std::vector<float> time_;

    static float mean(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::reduce(v.begin(), v.end()) / count;
    }

    static float median(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;
        else {
            sort(v.begin(), v.end());
            if (size % 2 == 0)
                return (v[size / 2 - 1] + v[size / 2]) / 2;
            else
                return v[size / 2];
        }
    }

    static float min(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.front();
    }

    static float max(std::vector<float> v) {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v.back();
    }

    static double std_dev(std::vector<float> const &v) {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::reduce(v.begin(), v.end()) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / count);
    }
};


//Newly Added When Implementing Bootstrapping

/*
 * Subroutines used by the linear transformation homomorphic capability
 */

namespace phantom {

struct longDiv {
    std::vector<double> q;
    std::vector<double> r;

    longDiv() {}
    longDiv(const std::vector<double>& q0, const std::vector<double>& r0) : q(q0), r(r0) {}
};

/**
 * @brief Gets the degree of a polynomial specified by its coefficients, which is the index of
 * the last non-zero element in the coefficients. If all the coefficients are zero, it returns 0.
 * @param coefficients vector of coefficients of a polynomial (can not be empty)
 * @return the integer degree of the polynomial.
 */
uint32_t Degree(const std::vector<double>& coefficients);

/**
 * Computes the quotient and remainder of the long division of two polynomials in the power series basis.
 *
 * @param &f the vector of coefficients of the dividend.
 * @param &g the vector of coefficients of the divisor.
 * @return a struct with the coefficients for the quotient and remainder.
 */
std::shared_ptr<longDiv> LongDivisionPoly(const std::vector<double>& f, const std::vector<double>& g);

/**
 * Computes the quotient and remainder of the long division of two polynomials in the Chebyshev series basis
 *
 * @param &f the vector of coefficients of the dividend.
 * @param &g the vector of coefficients of the divisor.
 * @return a struct with the coefficients for the quotient and remainder.
 */
std::shared_ptr<longDiv> LongDivisionChebyshev(const std::vector<double>& f, const std::vector<double>& g);

/**
 * Computes the values of the internal degrees k and m needed in the Paterson-Stockmeyer algorithm
 * such that k(2^m - 1} > n and k close to sqrt(n/2).
 *
 * @param n the degree of a polynomial.
 * @return a vector containing k and m.
 */
std::vector<uint32_t> ComputeDegreesPS(const uint32_t n);

/**
 * Get the depth for a given vector of coefficients for the Paterson-Stockmeyer algorithm.
 * The functions is based on the table described in src/pke/examples/FUNCTION_EVALUATION.md
 *
 * @param vec vector of coefficients
 * @param isNormalized true if the vector normalized. false is the default value
 * @return multiplicative depth
 */
uint32_t GetMultiplicativeDepthByCoeffVector(const std::vector<double>& vec, bool isNormalized = false);

/**
 * Extracts shifted diagonal of matrix A.
 *
 * @param &A square linear map.
 * @param index the index by which the diagonal shifted.
 *
 * @return the vector corresponding to the shifted diagonal
 */
std::vector<std::complex<double>> ExtractShiftedDiagonal(const std::vector<std::vector<std::complex<double>>>& A,
                                                         int index);

/**
 * Rotates a vector by an index - left rotation
 *
 * @param &a square linear map.
 * @param index rotation index.
 *
 * @return the rotated vector
 */
 std::vector<cuDoubleComplex> Rotate(const std::vector<std::complex<double>>& a, int32_t index);

/**
 * Clones the current vector up to the size indicated by the "slote" variable
 *
 * @param &a square linear map.
 * @param slots the new size of the vector.
 *
 * @return the vector with cloned values
 */
std::vector<std::complex<double>> Fill(const std::vector<std::complex<double>>& a, int slots);

/**
 * Computes the coefficients for the FFT encoding for CoeffEncodingCollapse such that every
 * iteration occupies one level.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
 */
std::vector<std::vector<std::complex<double>>> CoeffEncodingOneLevel(const std::vector<std::complex<double>>& pows,
                                                                     const std::vector<uint32_t>& rotGroup,
                                                                     bool flag_i);

/**
 * Computes the coefficients for the FFT decoding for CoeffDecodingCollapse such that every
 * iteration occupies one level.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
 */
std::vector<std::vector<std::complex<double>>> CoeffDecodingOneLevel(const std::vector<std::complex<double>>& pows,
                                                                     const std::vector<uint32_t>& rotGroup,
                                                                     bool flag_i);

/**
 * Computes the coefficients for the given level budget for the FFT encoding. Needed in
 * EvalLTFFTPrecomputeEncoding.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param levelBudget the user specified budget for levels.
 * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
 */
std::vector<std::vector<std::vector<std::complex<double>>>> CoeffEncodingCollapse(
    const std::vector<std::complex<double>>& pows, const std::vector<uint32_t>& rotGroup, uint32_t levelBudget,
    bool flag_i);

/**
 * Computes the coefficients for the given level budget for the FFT decoding. Needed in
 * EvalLTFFTPrecomputeDecoding.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute FFT.
 * @param levelBudget the user specified budget for levels.
 * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
 */
std::vector<std::vector<std::vector<std::complex<double>>>> CoeffDecodingCollapse(
    const std::vector<std::complex<double>>& pows, const std::vector<uint32_t>& rotGroup, uint32_t levelBudget,
    bool flag_i);

/**
 * Ensures that the index for rotation is positive and between 1 and slots.
 *
 * @param index signed rotation amount.
 * @param slots number of slots and size of vector that is rotated.
 */
uint32_t ReduceRotation(int32_t index, uint32_t slots);

/**
 * Computes parameters to ensure the encoding and decoding computations take exactly the
 * specified number of levels. More specifically, it returns a vector than contains
 * layers (the number of layers to collapse in one level), rows (how many such levels),
 * rem (the number of layers remaining to be collapsed in one level)
 *
 * @param logSlots the base 2 logarithm of the number of slots.
 * @param budget the allocated level budget for the computation.
 */
std::vector<uint32_t> SelectLayers(uint32_t logSlots, uint32_t budget = 4);

/**
 * Computes all parameters needed for the homomorphic encoding and decoding in the bootstrapping
 * operation and returns them as a vector. The returned vector's data can be accessed using
 * enum'ed indices from CKKS_BOOT_PARAMS that are defined below.
 *
 * @param slots number of slots
 * @param levelBudget the allocated level budget for the computation.
 * @param dim1 the value for the inner dimension in the baby-step giant-step strategy
 * @return vector with parameters for the homomorphic encoding and decoding in bootstrapping
 */
std::vector<int32_t> GetCollapsedFFTParams(uint32_t slots, uint32_t levelBudget = 4, uint32_t dim1 = 0);

/**
 *  Gets inner loop dimension for baby step giant step algorithm for linear transform,
 * taking into account the cost efficiency of hoisted automorphisms.
 * @param slots number of slots.
 * @return the value for the inner dimension in the baby-step giant-step strategy
*/
uint32_t getRatioBSGSLT(uint32_t slots);

/**
 * Assembles a list of rotation indices necessary to perform the
 * linear transform in scheme switching (needs to be ran once to each LT).
 * @param dim1 baby-step dimension
 * @param m cyclotomic order
 * @param blockdimension dimension related to the linear transform computation matrix
 * @return vector of rotation indices necessary
*/
std::vector<int32_t> FindLTRotationIndicesSwitch(uint32_t dim1, uint32_t m, uint32_t blockDimension);

/**
 * Assembles a list of rotation indices necessary to perform all the
 * linear transforms in argmin.
 * @param m cyclotomic order
 * @param blockdimension dimension related to the linear transform computation matrix
 * @param cols dimension of columns of the linear transform
 * @return vector of rotation indices necessary
*/
std::vector<int32_t> FindLTRotationIndicesSwitchArgmin(uint32_t m, uint32_t blockDimension, uint32_t cols);


// Reverse bits using CUDA intrinsics for performance
__device__ inline uint32_t reverse_bits_aux(uint32_t operand) {
    return __brev(operand);  // CUDA intrinsic for bit reversal (fast)
}

// Reverse bits for 64-bit integers using CUDA intrinsic
__device__ inline uint64_t reverse_bits_aux(uint64_t operand) {
    return __brevll(operand);  // CUDA intrinsic for 64-bit bit reversal
}

// Reverse only the first 'bit_count' bits
template<typename T>
__device__ inline T reverse_bits_aux(T operand, int bit_count) {
    return (bit_count == 0)
        ? T(0)
        : reverse_bits_aux(operand) >> (sizeof(T) * 8 - bit_count);
}

bool IsNotEqualOne(double val); 

uint32_t FindAutomorphismIndex2nComplex(int32_t i, uint32_t m);

__global__ void PrecomputeAutoMapKernel(uint32_t n, uint32_t k, uint32_t* precomp);

namespace CKKS_BOOT_PARAMS {
/**
   * Enums representing indices for the vector returned by GetCollapsedFFTParams()
   */
enum {
    LEVEL_BUDGET,  // the level budget
    LAYERS_COLL,   // the number of layers to collapse in one level
    LAYERS_REM,  // the number of layers remaining to be collapsed in one level to have exactly the number of levels specified in the level budget
    NUM_ROTATIONS,      // the number of rotations in one level
    BABY_STEP,          // the baby step in the baby-step giant-step strategy
    GIANT_STEP,         // the giant step in the baby-step giant-step strategy
    NUM_ROTATIONS_REM,  // the number of rotations in the remaining level
    BABY_STEP_REM,      // the baby step in the baby-step giant-step strategy for the remaining level
    GIANT_STEP_REM,     // the giant step in the baby-step giant-step strategy for the remaining level
    TOTAL_ELEMENTS      // total number of elements in the vector
};
}  // namespace CKKS_BOOT_PARAMS

}  // namespace lbcrypto


