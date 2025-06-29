#pragma once

#include <cmath>

#include "ciphertext.h"
#include "context.cuh"
#include "ntt.cuh"
#include "plaintext.h"
#include "secretkey.h"
#include "cuda_wrapper.cuh"
#include "./host/rns.h"

namespace phantom
{

    size_t
    FindLevelsToDrop(const PhantomContext &context, size_t multiplicativeDepth, double dcrtBits, bool isKeySwitch,
                     bool isAsymmetric);

    __global__ void key_switch_inner_prod_c2_and_evk(uint64_t *dst, const uint64_t *c2, const uint64_t *const *evks,
                                                     const DModulus *modulus, size_t n, size_t size_QP,
                                                     size_t size_QP_n,
                                                     size_t size_QlP, size_t size_QlP_n, size_t size_Q, size_t size_Ql,
                                                     size_t beta, size_t reduction_threshold);

    // used by keyswitch_inplace
    void key_switch_inner_prod(uint64_t *p_cx, const uint64_t *p_t_mod_up, const uint64_t *const *rlk,
                               const phantom::DRNSTool &rns_tool, const DModulus *modulus_QP,
                               size_t reduction_threshold, const cudaStream_t &stream);

    void keyswitch_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, uint64_t *c2,
                           const PhantomRelinKey &relin_keys,
                           bool is_relin, // false
                           const cudaStream_t &stream);

    /***************************************************** Core APIs ******************************************************/

    // encrypted = -encrypted
    void negate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted);

    inline auto negate(const PhantomContext &context, const PhantomCiphertext &encrypted)
    {
        PhantomCiphertext destination = encrypted;
        negate_inplace(context, destination);
        return destination;
    }

    // encrypted1 += encrypted2
    void add_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    inline auto
    add(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2)
    {
        PhantomCiphertext destination = encrypted1;
        add_inplace(context, destination, encrypted2);
        return destination;
    }

    // destination = encrypteds[0] + encrypteds[1] + ...
    void add_many(const PhantomContext &context, const std::vector<PhantomCiphertext> &encrypteds,
                  PhantomCiphertext &destination);

    // if negate = false (default): encrypted1 -= encrypted2
    // if negate = true: encrypted1 = encrypted2 - encrypted1
    void sub_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
                     const bool &negate = false);

    inline auto
    sub(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2,
        const bool &negate = false)
    {
        PhantomCiphertext destination = encrypted1;
        sub_inplace(context, destination, encrypted2, negate);
        return destination;
    }

    // encrypted += plain
    void add_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    add_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain)
    {
        PhantomCiphertext destination = encrypted;
        add_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted -= plain
    void sub_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto
    sub_plain(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain)
    {
        PhantomCiphertext destination = encrypted;
        sub_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted *= plain
    void
    multiply_plain_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, const PhantomPlaintext &plain);

    inline auto multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted,
                               const PhantomPlaintext &plain)
    {
        PhantomCiphertext destination = encrypted;
        multiply_plain_inplace(context, destination, plain);
        return destination;
    }

    // encrypted1 *= encrypted2
    void
    multiply_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);

    inline auto
    multiply(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2)
    {
        PhantomCiphertext destination = encrypted1;
        multiply_inplace(context, destination, encrypted2);
        return destination;
    }

    // encrypted1 *= encrypted2
    void multiply_and_relin_inplace(const PhantomContext &context, PhantomCiphertext &encrypted1,
                                    const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys);

    inline auto multiply_and_relin(const PhantomContext &context, const PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys)
    {
        PhantomCiphertext destination = encrypted1;
        multiply_and_relin_inplace(context, destination, encrypted2, relin_keys);
        return destination;
    }

    void relinearize_inplace(const PhantomContext &context, PhantomCiphertext &encrypted,
                             const PhantomRelinKey &relin_keys);

    inline auto relinearize(const PhantomContext &context, const PhantomCiphertext &encrypted,
                            const PhantomRelinKey &relin_keys)
    {
        PhantomCiphertext destination = encrypted;
        relinearize_inplace(context, destination, relin_keys);
        return destination;
    }

    // ciphertext
    [[nodiscard]]
    PhantomCiphertext rescale_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    // ciphertext
    inline void rescale_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted)
    {
        encrypted = rescale_to_next(context, encrypted);
    }

    // ciphertext
    [[nodiscard]]
    PhantomCiphertext mod_switch_to_next(const PhantomContext &context, const PhantomCiphertext &encrypted);

    // ciphertext
    inline void mod_switch_to_next_inplace(const PhantomContext &context, PhantomCiphertext &encrypted)
    {
        encrypted = mod_switch_to_next(context, encrypted);
    }

    // ciphertext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t chain_index)
    {
        if (encrypted.chain_index() > chain_index)
        {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomCiphertext destination = encrypted;

        while (destination.chain_index() != chain_index)
        {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    // ciphertext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t chain_index)
    {
        if (encrypted.chain_index() > chain_index)
        {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.chain_index() != chain_index)
        {
            mod_switch_to_next_inplace(context, encrypted);
        }
    }

    // plaintext
    void mod_switch_to_next_inplace(const PhantomContext &context, PhantomPlaintext &plain);

    // plaintext
    inline auto mod_switch_to_next(const PhantomContext &context, const PhantomPlaintext &plain)
    {
        PhantomPlaintext destination = plain;
        mod_switch_to_next_inplace(context, destination);
        return destination;
    }

    // plaintext
    inline void mod_switch_to_inplace(const PhantomContext &context, PhantomPlaintext &plain, size_t chain_index)
    {
        if (plain.chain_index() > chain_index)
        {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.chain_index() != chain_index)
        {
            mod_switch_to_next_inplace(context, plain);
        }
    }

    // plaintext
    inline auto mod_switch_to(const PhantomContext &context, const PhantomPlaintext &plain, size_t chain_index)
    {
        if (plain.chain_index() > chain_index)
        {
            throw std::invalid_argument("cannot switch to higher level modulus");
        }

        PhantomPlaintext destination = plain;

        while (destination.chain_index() != chain_index)
        {
            mod_switch_to_next_inplace(context, destination);
        }

        return destination;
    }

    void apply_galois_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t galois_elt,
                              const PhantomGaloisKey &galois_keys);

    inline auto apply_galois(const PhantomContext &context, const PhantomCiphertext &encrypted, size_t galois_elt,
                             const PhantomGaloisKey &galois_keys)
    {
        PhantomCiphertext destination = encrypted;
        apply_galois_inplace(context, destination, galois_elt, galois_keys);
        return destination;
    }

    void rotate_inplace(const PhantomContext &context, PhantomCiphertext &encrypted, int step,
                        const PhantomGaloisKey &galois_key);

    inline auto rotate(const PhantomContext &context, const PhantomCiphertext &encrypted, int step,
                       const PhantomGaloisKey &galois_key)
    {
        PhantomCiphertext destination = encrypted;
        rotate_inplace(context, destination, step, galois_key);
        return destination;
    }

    /***************************************************** Newly Added (BootStrapping) ******************************************************/
    // The functions here work same as FLEXIBLEAUTO option in OPENFHE

    void EvalRotateFused(const PhantomContext &context, const PhantomGaloisKeyFused &galois_keys, PhantomCiphertext &in_ciphertext, PhantomCiphertext &out_ciphertext, int32_t index);

    void EvalConjFused(const PhantomContext &context, const PhantomGaloisKeyFused &galois_keys, PhantomCiphertext &in_ciphertext, PhantomCiphertext &out_ciphertext);

    PhantomCiphertext ModReduce(const PhantomContext &context, PhantomCiphertext &ciphertext, size_t levels);

    inline void EvalModReduceInPlace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t levels)
    {
        encrypted = ModReduce(context, encrypted, levels);
    }

    std::vector<uint64_t> GetElementForEvalMult(const PhantomContext &context, PhantomCiphertext &ciphertext, double operand, const std::vector<double> &m_scalingFactorsReal);

    inline void CRTMult(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b,
                        const std::vector<arith::Modulus> &mods, std::vector<uint64_t> &result)
    {

        for (uint32_t i = 0; i < a.size(); i++)
        {
            result[i] = multiply_uint_mod(a[i], b[i], mods[i]);
        }
    }

    PhantomCiphertext RaiseMod(const PhantomContext &context, PhantomCiphertext &ciphertext, bool transform = true);

    __global__ void switchModulusKernel(const uint64_t *operand, uint64_t *result, const DModulus *modulus, const uint64_t poly_degree, const uint64_t coeff_mod_size);

    __global__ void init_monomial_kernel(uint64_t *monomial, uint32_t index, uint32_t powerReduced, uint32_t size_Ql_n, uint32_t N, const DModulus *moduli);

    void MultByMonomialInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext, uint32_t power);

    void ConvertToEval(const PhantomContext &context, PhantomCiphertext &ciphertext);

    void ConvertToCoeff(const PhantomContext &context, PhantomCiphertext &ciphertext);

    void AdjustLevelsAndDepthInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext1, PhantomCiphertext &ciphertext2, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    // ciphertext
    inline void ModSwitchLevelInPlace(const PhantomContext &context, PhantomCiphertext &encrypted, size_t levels)
    {
        size_t noise = encrypted.GetNoiseScaleDeg();
        for (uint32_t i = 0; i < levels; i++)
        {
            mod_switch_to_next_inplace(context, encrypted);
        }
        encrypted.SetNoiseScaleDeg(noise);
    }

    // encrypted1 *= c
    void EvalMultConstInplaceCore(const PhantomContext &context, PhantomCiphertext &encrypted1, const double scalar, const std::vector<double> &m_scalingFactorsReal);

    inline void EvalMultConstInplace(const PhantomContext &context, PhantomCiphertext &encrypted1, const double scalar, const std::vector<double> &m_scalingFactorsReal)
    {

        if (encrypted1.GetNoiseScaleDeg() == 2)
        {
            EvalModReduceInPlace(context, encrypted1, 1);
        }

        EvalMultConstInplaceCore(context, encrypted1, scalar, m_scalingFactorsReal);
    }

    inline PhantomCiphertext EvalMultConstCore(const PhantomContext &context, const PhantomCiphertext &encrypted1, const double scalar, const std::vector<double> &m_scalingFactorsReal)
    {

        PhantomCiphertext destination = encrypted1;
        EvalMultConstInplace(context, destination, scalar, m_scalingFactorsReal);
        return destination;
    }

    inline PhantomCiphertext EvalMultConst(const PhantomContext &context, const PhantomCiphertext &encrypted1, const double scalar, const std::vector<double> &m_scalingFactorsReal)
    {

        PhantomCiphertext result = encrypted1;
        EvalMultConstInplace(context, result, scalar, m_scalingFactorsReal);
        return result;
    }

    PhantomCiphertext EvalMultAuto(const PhantomContext &context, const PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2,
                                   const PhantomRelinKey &relin_keys, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    PhantomCiphertext KeySwitchDown(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    PhantomCiphertext KeySwitchDownFirstElement(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    std::vector<uint64_t> GetElementForEvalAddOrSub(const PhantomContext &context, const PhantomCiphertext &ciphertext, const double operand,
                                                    const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    void EvalAddConstInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext, const double operand,
                             const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    void EvalSubConstInPlace(const PhantomContext &context, const PhantomCiphertext &ciphertext, double operand,
                             const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    PhantomCiphertext EvalChebyshevSeries(const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomCiphertext &x,
                                          const std::vector<double> &coefficients, double a, double b, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    PhantomCiphertext EvalChebyshevSeriesLinear(const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomCiphertext &x,
                                                const std::vector<double> &coefficients, double a, double b, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    PhantomCiphertext EvalChebyshevSeriesPS(const PhantomContext &context, const PhantomRelinKey &relin_keys, const PhantomCiphertext &x,
                                            const std::vector<double> &coefficients, double a, double b, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    std::shared_ptr<longDiv> LongDivisionPoly(const std::vector<double> &f, const std::vector<double> &g);

    PhantomCiphertext EvalLinearWSumMutable(const PhantomContext &context, std::vector<std::shared_ptr<PhantomCiphertext>> &ciphertexts,
                                            const std::vector<double> &constants, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    PhantomCiphertext InnerEvalChebyshevPS(const PhantomContext &context, const PhantomCiphertext &x, const PhantomRelinKey &relin_keys,
                                           const std::vector<double> &coefficients, uint32_t k, uint32_t m,
                                           std::vector<std::shared_ptr<PhantomCiphertext>> &T, std::vector<std::shared_ptr<PhantomCiphertext>> &T2,
                                           const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    std::vector<double> EvalChebyshevCoefficients(std::function<double(double)> func, double a, double b, uint32_t degree);

    inline PhantomCiphertext EvalChebyshevFunction(std::function<double(double)> func, const PhantomContext &context, const PhantomRelinKey &relin_keys,
                                                   PhantomCiphertext ciphertext, double a, double b, uint32_t degree,
                                                   const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {

        std::vector<double> coefficients = EvalChebyshevCoefficients(func, a, b, degree);
        return EvalChebyshevSeries(context, relin_keys, ciphertext, coefficients, a, b, m_scalingFactorsReal, m_scalingFactorsRealBig);
    }

    void EvalAddAutoInplace(const PhantomContext &context, PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2,
                            const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    void EvalSubAutoInplace(const PhantomContext &context, PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2,
                            const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    inline PhantomCiphertext EvalAddAuto(const PhantomContext &context, const PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2,
                                         const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {

        PhantomCiphertext destination = ciphertext1;

        EvalAddAutoInplace(context, destination, ciphertext2, m_scalingFactorsReal, m_scalingFactorsRealBig);
        return destination;
    }

    inline PhantomCiphertext EvalSubAuto(const PhantomContext &context, const PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2,
                                         const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {

        PhantomCiphertext destination = ciphertext1;

        EvalSubAutoInplace(context, destination, ciphertext2, m_scalingFactorsReal, m_scalingFactorsRealBig);
        return destination;
    }

    PhantomCiphertext EvalSquare(const PhantomContext &context, const PhantomCiphertext &ciphertext1,
                                 const PhantomRelinKey &relin_keys, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    inline void EvalSquareInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext1,
                                  const PhantomRelinKey &relin_keys, const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {
        ciphertext1 = EvalSquare(context, ciphertext1, relin_keys, m_scalingFactorsReal, m_scalingFactorsRealBig);
    }

    inline void EvalAddConstInPlaceWrap(const PhantomContext &context, PhantomCiphertext &ciphertext, const double operand,
                                        const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {

        if (operand == 0)
        {
            return;
        }

        if (operand > 0)
        {
            EvalAddConstInPlace(context, ciphertext, operand, m_scalingFactorsReal, m_scalingFactorsRealBig);
        }

        else
        {
            EvalSubConstInPlace(context, ciphertext, -operand, m_scalingFactorsReal, m_scalingFactorsRealBig);
        }
    }

    inline PhantomCiphertext EvalAddConst(const PhantomContext &context, const PhantomCiphertext &ciphertext, double operand,
                                          const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig)
    {
        PhantomCiphertext destination = ciphertext;
        EvalAddConstInPlaceWrap(context, destination, operand, m_scalingFactorsReal, m_scalingFactorsRealBig);
        return destination;
    }

    util::cuda_auto_ptr<uint64_t> EvalFastRotationPrecompute(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    PhantomCiphertext EvalFastRotationExt(const PhantomContext &context, const PhantomCiphertext &ciphertext, const PhantomGaloisKeyFused &galois_keys,
                                          int32_t index, util::cuda_auto_ptr<uint64_t> digits, bool add_first);

    void EvalMultExtInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext, const PhantomPlaintext &plaintext);

    inline PhantomCiphertext EvalMultExt(const PhantomContext &context, const PhantomCiphertext &ciphertext, const PhantomPlaintext &plaintext)
    {

        PhantomCiphertext destination = ciphertext;
        EvalMultExtInPlace(context, destination, plaintext);

        return destination;
    }

    void EvalAddExtInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2);

    inline PhantomCiphertext EvalAddExt(const PhantomContext &context, const PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2)
    {

        PhantomCiphertext destination = ciphertext1;
        EvalAddExtInPlace(context, destination, ciphertext2);

        return destination;
    }

    PhantomCiphertext KeySwitchExt(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    void MultByIntegerInPlace(const PhantomContext &context, PhantomCiphertext &ciphertext, uint64_t integer);

    void add_two_poly_inplace(const PhantomContext &context, uint64_t *poly1, const uint64_t *poly2, size_t chain_index);

    util::cuda_auto_ptr<uint64_t> rotate_c0(const PhantomContext &context, const PhantomCiphertext &ciphertext, size_t rot_idx);

    void add_two_poly_inplace_ext(const PhantomContext &context, uint64_t *poly1, const uint64_t *poly2, size_t chain_index);

    void reset_poly_ext(const PhantomContext &context, uint64_t *poly1, size_t chain_index);

    void EvalMultBroadcast(const PhantomContext &context, PhantomCiphertext &ciphertext1, const PhantomCiphertext &ciphertext2);

    void EvalMultAutoInplace(const PhantomContext &context, PhantomCiphertext &ciphertext, const PhantomPlaintext &plaintext,
                             const std::vector<double> &m_scalingFactorsReal, const std::vector<double> &m_scalingFactorsRealBig);

    /*************************************************** Advanced APIs ****************************************************/

    void hoisting_inplace(const PhantomContext &context, PhantomCiphertext &ct, const PhantomGaloisKey &glk,
                          const std::vector<int> &steps);

    inline auto hoisting(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomGaloisKey &glk,
                         const std::vector<int> &steps)
    {
        PhantomCiphertext destination = encrypted;
        hoisting_inplace(context, destination, glk, steps);
        return destination;
    }

}
