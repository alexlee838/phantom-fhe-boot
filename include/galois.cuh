#pragma once

#include <cstdint>
#include <vector>

#include "host/modulus.h"
#include "host/defines.h"
#include "host/uintcore.h"
#include "cuda_wrapper.cuh"
#include "ntt.cuh"
#include "util.cuh"

namespace phantom::util {

    constexpr uint32_t generator_ = 5;

    [[nodiscard]] inline auto get_elt_from_step(int step, size_t coeff_count) {
        auto n = static_cast<uint32_t>(coeff_count);
        uint32_t m32 = n * 2;
        auto m = static_cast<uint64_t>(m32);

        if (step == 0) {
            return static_cast<uint32_t>(m - 1);
        } else {
            // Extract sign of steps. When steps is positive, the rotation
            // is to the left; when steps is negative, it is to the right.
            bool sign = step < 0;
            auto pos_step = static_cast<uint32_t>(abs(step));

            if (pos_step >= (n >> 1)) {
                throw std::invalid_argument("step count too large");
            }

            pos_step &= m32 - 1;
            if (sign) {
                step = static_cast<int>(n >> 1) - static_cast<int>(pos_step);
            } else {
                step = static_cast<int>(pos_step);
            }

            // Construct Galois element for row rotation
            auto gen = static_cast<uint64_t>(generator_);
            uint64_t galois_elt = 1;
            while (step--) {
                galois_elt *= gen;
                galois_elt &= m - 1;
            }
            return static_cast<uint32_t>(galois_elt);
        }
    }

    [[nodiscard]] inline auto get_elts_from_steps(const std::vector<int> &steps, size_t coeff_count) {
        std::vector<std::uint32_t> galois_elts;
        for (auto step: steps) {
            galois_elts.push_back(get_elt_from_step(step, coeff_count));
        }
        return galois_elts;
    }

    class PhantomGaloisTool {

    private:

        int coeff_count_power_ = 0;
        std::size_t coeff_count_ = 0;
        std::vector<uint32_t> galois_elts_{};
        std::vector<phantom::util::cuda_auto_ptr<uint32_t>> permutation_tables_;
        std::vector<phantom::util::cuda_auto_ptr<uint64_t>> index_raw_tables_; // only used by bfv
        bool is_bfv_;

        /**
        Compute a vector of all necessary galois_elts.
        */
        [[nodiscard]] std::vector<uint32_t> get_elts_all() const;

    public:

        explicit PhantomGaloisTool(const std::vector<uint32_t> &galois_elts, int coeff_count_power,
                                   const cudaStream_t &stream, bool is_bfv = false) {
            if ((coeff_count_power < phantom::arith::get_power_of_two(POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > phantom::arith::get_power_of_two(POLY_MOD_DEGREE_MAX)) {
                throw std::invalid_argument("coeff_count_power out of range");
            }
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;

            // if user has not provided galois_elts, compute all power of 2 galois_elts to construct NAF rotation
            if (galois_elts.empty()) {
                galois_elts_ = get_elts_all();
            } else {
                galois_elts_ = galois_elts;
            }

            is_bfv_ = is_bfv;

            auto galois_elts_size = galois_elts_.size();
            const auto coeff_count_minus_one = static_cast<uint32_t>(coeff_count_) - 1;

            // compute permutation_tables_
            permutation_tables_.resize(galois_elts_size);
            std::vector<uint32_t> u32temp(coeff_count_);
            for (std::size_t idx{0}; idx < galois_elts_size; idx++) {
                permutation_tables_[idx] = phantom::util::make_cuda_auto_ptr<uint32_t>(coeff_count_, stream);
                auto galois_elt = galois_elts_.at(idx);
                auto temp_ptr = u32temp.data();
                for (size_t i = coeff_count_; i < coeff_count_ << 1; i++) {
                    uint32_t reversed = phantom::arith::reverse_bits(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                    uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                    index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                    *temp_ptr++ = phantom::arith::reverse_bits(static_cast<uint32_t>(index_raw), coeff_count_power_);
                }
                cudaMemcpyAsync(permutation_tables_[idx].get(), u32temp.data(), coeff_count_ * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream);
            }

            if (is_bfv_) {
                index_raw_tables_.resize(galois_elts_size);
                std::vector<uint64_t> u64temp(coeff_count_);
                for (std::size_t idx = 0; idx < galois_elts_size; idx++) {
                    index_raw_tables_[idx] = phantom::util::make_cuda_auto_ptr<uint64_t>(coeff_count_, stream);
                    auto galois_elt = galois_elts_.at(idx);
                    auto temp_ptr = u64temp.data();
                    uint64_t index_raw = 0;
                    for (uint64_t i = 0; i <= coeff_count_minus_one; i++) {
                        *temp_ptr++ = index_raw;
                        index_raw = (index_raw + galois_elt) & ((coeff_count_ << 1) - 1); // (mod 2n-1)
                    }
                    cudaMemcpyAsync(index_raw_tables_[idx].get(), u64temp.data(), coeff_count_ * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, stream);
                }
            }
        }

        PhantomGaloisTool(const PhantomGaloisTool &copy) = delete;

        PhantomGaloisTool &operator=(const PhantomGaloisTool &copy) = delete;

        PhantomGaloisTool(PhantomGaloisTool &&move) = delete;

        PhantomGaloisTool &operator=(PhantomGaloisTool &&move) = delete;

        ~PhantomGaloisTool() = default;

        [[nodiscard]] auto &galois_elts() const {
            return galois_elts_;
        }

        [[nodiscard]] inline std::vector<std::uint32_t> get_elts_from_steps(const std::vector<int> &steps) const {
            std::vector<std::uint32_t> elts;
            for (auto step: steps)
                elts.push_back(get_elt_from_step(step, coeff_count_));
            return elts;
        }

        void apply_galois(uint64_t *operand, const DNTTTable &rns_table, size_t coeff_mod_size, size_t galois_elt_idx,
                          uint64_t *result, const cudaStream_t &stream);

        void apply_galois_ntt(uint64_t *operand, size_t coeff_mod_size, size_t galois_elt_idx, uint64_t *result,
                              const cudaStream_t &stream);

        void apply_galois_ntt_direct(uint64_t *operand0,  size_t coeff_mod_size,
                                    uint64_t *result0, uint32_t *d_vec, const cudaStream_t &stream);


    };
    __global__ void apply_galois_ntt_permutation_direct(uint64_t *dst, const uint64_t *src, const uint32_t *permutation_table,
                                                 size_t poly_degree, uint64_t coeff_mod_size);
}
