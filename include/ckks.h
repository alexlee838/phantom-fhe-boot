#pragma once

#include <cuComplex.h>
#include <optional>

#include "context.cuh"
#include "fft.h"
#include "ntt.cuh"
#include "plaintext.h"
#include "rns.cuh"

class PhantomCKKSEncoder
{

private:
    uint32_t slots_{};
    std::unique_ptr<phantom::util::ComplexRoots> complex_roots_;
    std::vector<cuDoubleComplex> root_powers_;
    std::vector<uint32_t> rotation_group_;
    std::unique_ptr<DCKKSEncoderInfo> gpu_ckks_msg_vec_;

    std::unique_ptr<DCKKSEncoderInfo> sparse_gpu_ckks_msg_vec_;
    std::unique_ptr<phantom::util::ComplexRoots> sparse_complex_roots_;
    std::vector<cuDoubleComplex> sparse_root_powers_;
    std::vector<uint32_t> sparse_rotation_group_;

    std::optional<PhantomContext> sparse_context_;

    std::unique_ptr<DCKKSEncoderInfo> sparse_bootstrap_gpu_ckks_msg_vec_;
    std::unique_ptr<phantom::util::ComplexRoots> sparse_bootstrap_complex_roots_;
    std::vector<cuDoubleComplex> sparse_bootstrap_root_powers_;
    std::vector<uint32_t> sparse_bootstrap_rotation_group_;

    std::optional<PhantomContext> sparse_bootstrap_context_;

    uint32_t first_chain_index_ = 1;

    void encode_internal(const PhantomContext &context,
                         const std::vector<cuDoubleComplex> &values,
                         size_t chain_index, double scale,
                         PhantomPlaintext &destination,
                         const cudaStream_t &stream);

    inline void encode_internal(const PhantomContext &context,
                                const std::vector<double> &values,
                                size_t chain_index, double scale,
                                PhantomPlaintext &destination,
                                const cudaStream_t &stream)
    {
        size_t values_size = values.size();
        std::vector<cuDoubleComplex> input(values_size);
        for (size_t i = 0; i < values_size; i++)
        {
            input[i] = make_cuDoubleComplex(values[i], 0.0);
        }
        encode_internal(context, input, chain_index, scale, destination, stream);
    }

    inline void encode_internal(const PhantomContext &context,
                                const std::vector<std::complex<double>> &values,
                                size_t chain_index, double scale,
                                PhantomPlaintext &destination,
                                const cudaStream_t &stream)
    {
        size_t values_size = values.size();
        std::vector<cuDoubleComplex> input(values_size);

        for (size_t i = 0; i < values_size; i++)
        {
            input[i] = make_cuDoubleComplex(values[i].real(), values[i].imag());
        }

        encode_internal(context, input, chain_index, scale, destination, stream);
    }

    void encode_internal_ext(const PhantomContext &context,
                             const std::vector<cuDoubleComplex> &values,
                             size_t chain_index, double scale,
                             PhantomPlaintext &destination,
                             const cudaStream_t &stream);

    void encode_sparse_internal(const PhantomContext &context, const std::vector<cuDoubleComplex> &values,
                                size_t chain_index, double scale, PhantomPlaintext &destination, const cudaStream_t &stream);

    void encode_sparse_internal_ext(const PhantomContext &context, const std::vector<cuDoubleComplex> &values,
                                    size_t chain_index, double scale, PhantomPlaintext &destination, const cudaStream_t &stream);

    inline void encode_sparse_internal(const PhantomContext &context,
                                       const std::vector<double> &values,
                                       size_t chain_index, double scale,
                                       PhantomPlaintext &destination,
                                       const cudaStream_t &stream)
    {
        size_t values_size = values.size();
        std::vector<cuDoubleComplex> input(values_size);
        for (size_t i = 0; i < values_size; i++)
        {
            input[i] = make_cuDoubleComplex(values[i], 0.0);
        }
        encode_sparse_internal(context, input, chain_index, scale, destination, stream);
    }

    inline void encode_sparse_internal(const PhantomContext &context,
                                       const std::vector<std::complex<double>> &values,
                                       size_t chain_index, double scale,
                                       PhantomPlaintext &destination,
                                       const cudaStream_t &stream)
    {
        size_t values_size = values.size();
        std::vector<cuDoubleComplex> input(values_size);

        for (size_t i = 0; i < values_size; i++)
        {
            input[i] = make_cuDoubleComplex(values[i].real(), values[i].imag());
        }

        encode_sparse_internal(context, input, chain_index, scale, destination, stream);
    }

    void decode_internal(const PhantomContext &context,
                         const PhantomPlaintext &plain,
                         std::vector<cuDoubleComplex> &destination,
                         const cudaStream_t &stream);

    inline void decode_internal(const PhantomContext &context,
                                const PhantomPlaintext &plain,
                                std::vector<double> &destination,
                                const cudaStream_t &stream)
    {
        std::vector<cuDoubleComplex> output;
        decode_internal(context, plain, output, stream);
        destination.resize(slots_);
        for (size_t i = 0; i < slots_; i++)
            destination[i] = output[i].x;
    }

    void decode_internal_ext(const PhantomContext &context,
                             const PhantomPlaintext &plain,
                             std::vector<cuDoubleComplex> &destination,
                             const cudaStream_t &stream);

    void decode_sparse_internal(const PhantomContext &context, const PhantomPlaintext &plain, size_t val_size,
                                std::vector<cuDoubleComplex> &destination, const cudaStream_t &stream);

    inline void decode_sparse_internal(const PhantomContext &context, const PhantomPlaintext &plain, size_t val_size,
                                       std::vector<double> &destination, const cudaStream_t &stream)
    {
        std::vector<cuDoubleComplex> output;
        decode_sparse_internal(context, plain, val_size, output, stream);
        destination.resize(val_size);
        for (size_t i = 0; i < val_size; i++)
            destination[i] = output[i].x;
    }

    void decode_sparse_internal_ext(const PhantomContext &context,
                                    const PhantomPlaintext &plain,
                                    size_t val_size,
                                    std::vector<cuDoubleComplex> &destination,
                                    const cudaStream_t &stream);

public:
    explicit PhantomCKKSEncoder(const PhantomContext &context);

    PhantomCKKSEncoder(const PhantomCKKSEncoder &copy) = delete;

    PhantomCKKSEncoder(PhantomCKKSEncoder &&source) = delete;

    PhantomCKKSEncoder &operator=(const PhantomCKKSEncoder &assign) = delete;

    PhantomCKKSEncoder &operator=(PhantomCKKSEncoder &&assign) = delete;

    ~PhantomCKKSEncoder() = default;

    template <class T>
    inline void encode(const PhantomContext &context,
                       const std::vector<T> &values,
                       double scale,
                       PhantomPlaintext &destination,
                       size_t chain_index = 1)
    { // first chain index

        const auto &s = cudaStreamPerThread;

        destination.chain_index_ = 0;

        destination.resize(context.coeff_mod_size_, context.poly_degree_, s);
        encode_internal(context, values, chain_index, scale, destination, s);
    }

    template <class T>
    inline void encode_ext(const PhantomContext &context,
                           const std::vector<T> &values,
                           double scale,
                           PhantomPlaintext &destination,
                           size_t chain_index)
    {

        const auto &s = cudaStreamPerThread;

        size_t size_Ql = context.get_context_data(chain_index).gpu_rns_tool().base_Ql().size();
        size_t size_P = context.get_context_data(0).parms().special_modulus_size();
        size_t size_QlP = size_Ql + size_P;

        destination.resize(size_QlP, context.poly_degree_, s);
        encode_internal_ext(context, values, chain_index, scale, destination, s);
    }

    template <class T>
    inline void encode_sparse_ext(const PhantomContext &context,
                                  const std::vector<T> &values,
                                  double scale,
                                  PhantomPlaintext &destination,
                                  size_t chain_index)
    {

        const auto &s = cudaStreamPerThread;

        size_t size_Ql = context.get_context_data(chain_index).gpu_rns_tool().base_Ql().size();
        size_t size_P = context.get_context_data(0).parms().special_modulus_size();
        size_t size_QlP = size_Ql + size_P;

        destination.resize(size_QlP, context.poly_degree_, s);
        encode_sparse_internal_ext(context, values, chain_index, scale, destination, s);
    }

    template <class T>
    [[nodiscard]] inline auto encode(const PhantomContext &context, const std::vector<T> &values,
                                     double scale,
                                     size_t chain_index = 1)
    { // first chain index

        PhantomPlaintext destination;
        encode(context, values, scale, destination, chain_index);
        return destination;
    }

    template <class T>
    [[nodiscard]] inline auto encode_ext(const PhantomContext &context, const std::vector<T> &values,
                                         double scale,
                                         size_t chain_index)
    { // first chain index

        PhantomPlaintext destination;
        encode_ext(context, values, scale, destination, chain_index);
        return destination;
    }

    template <class T>
    [[nodiscard]] inline auto encode_sparse_ext(const PhantomContext &context, const std::vector<T> &values,
                                                double scale,
                                                size_t chain_index)
    { // first chain index

        PhantomPlaintext destination;
        encode_sparse_ext(context, values, scale, destination, chain_index);
        return destination;
    }

    template <class T>
    inline void encode_sparse(const PhantomContext &context,
                              const std::vector<T> &values,
                              double scale,
                              PhantomPlaintext &destination,
                              size_t chain_index = 1)
    { // first chain index

        const auto &s = cudaStreamPerThread;

        destination.chain_index_ = 0;

        destination.resize(context.coeff_mod_size_, context.poly_degree_, s);
        encode_sparse_internal(context, values, chain_index, scale, destination, s);
    }

    template <class T>
    [[nodiscard]] inline auto encode_sparse(const PhantomContext &context, const std::vector<T> &values,
                                            double scale,
                                            size_t chain_index = 1)
    { // first chain index

        PhantomPlaintext destination;
        encode_sparse(context, values, scale, destination, chain_index);
        return destination;
    }

    template <class T>
    inline void decode(const PhantomContext &context,
                       const PhantomPlaintext &plain,
                       std::vector<T> &destination)
    {
        decode_internal(context, plain, destination, cudaStreamPerThread);
    }

    template <class T>
    inline void decode_ext(const PhantomContext &context,
                           const PhantomPlaintext &plain,
                           std::vector<T> &destination)
    {
        decode_internal(context, plain, destination, cudaStreamPerThread);
    }

    template <class T>
    inline void decode_sparse(const PhantomContext &context,
                              const PhantomPlaintext &plain,
                              size_t val_size,
                              std::vector<T> &destination)
    {
        decode_sparse_internal(context, plain, val_size, destination, cudaStreamPerThread);
    }

    template <class T>
    [[nodiscard]] inline auto decode_sparse(const PhantomContext &context, const PhantomPlaintext &plain, size_t val_size)
    {
        std::vector<T> destination;
        decode(context, plain, val_size, destination);
        return destination;
    }

    template <class T>
    [[nodiscard]] inline auto decode(const PhantomContext &context, const PhantomPlaintext &plain)
    {
        std::vector<T> destination;
        decode(context, plain, destination);
        return destination;
    }

    template <class T>
    [[nodiscard]] inline auto decode_ext(const PhantomContext &context, const PhantomPlaintext &plain)
    {
        std::vector<T> destination;
        decode(context, plain, destination);
        return destination;
    }

    [[nodiscard]] inline std::size_t slot_count() const noexcept
    {
        return slots_;
    }

    auto &gpu_ckks_msg_vec()
    {
        return *gpu_ckks_msg_vec_;
    }

    inline void set_sparse_encode(const phantom::EncryptionParameters &params, std::size_t coeff_count)
    {
        const auto &s = cudaStreamPerThread;

        uint32_t slots = coeff_count >> 1;
        uint32_t m = coeff_count << 1;
        uint32_t slots_half = slots >> 1;

        sparse_gpu_ckks_msg_vec_ = std::make_unique<DCKKSEncoderInfo>(coeff_count, s);

        // We need m powers of the primitive 2n-th root, m = 2n
        sparse_root_powers_.reserve(m);
        sparse_rotation_group_.reserve(slots_half);

        uint32_t gen = 5;
        uint32_t pos = 1; // Position in normal bit order
        for (size_t i = 0; i < slots_half; i++)
        {
            // Set the bit-reversed locations
            sparse_rotation_group_[i] = pos;

            // Next primitive root
            pos *= gen; // 5^i mod m
            pos &= (m - 1);
        }

        // Powers of the primitive 2n-th root have 4-fold symmetry
        if (m >= 8)
        {
            sparse_complex_roots_ = std::make_unique<phantom::util::ComplexRoots>(phantom::util::ComplexRoots(static_cast<size_t>(m)));
            for (size_t i = 0; i < m; i++)
            {
                sparse_root_powers_[i] = sparse_complex_roots_->get_root(i);
            }
        }
        else if (m == 4)
        {
            sparse_root_powers_[0] = {1, 0};
            sparse_root_powers_[1] = {0, 1};
            sparse_root_powers_[2] = {-1, 0};
            sparse_root_powers_[3] = {0, -1};
        }

        cudaMemcpyAsync(sparse_gpu_ckks_msg_vec_->twiddle(), sparse_root_powers_.data(), m * sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(sparse_gpu_ckks_msg_vec_->mul_group(), sparse_rotation_group_.data(), slots_half * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);

        auto temp_parms = params;
        temp_parms.set_poly_modulus_degree(coeff_count);
        sparse_context_.emplace(temp_parms); // Constructs PhantomContext in-place

        ///////////////////////////////////////////////
        /////////////// Sparse Bootstrap //////////////
        ///////////////////////////////////////////////

        uint32_t coeff_count_bs = coeff_count << 1;
        uint32_t slots_bs = coeff_count_bs >> 1;
        uint32_t m_bs = coeff_count_bs << 1;
        uint32_t slots_half_bs = slots_bs >> 1;

        sparse_bootstrap_gpu_ckks_msg_vec_ = std::make_unique<DCKKSEncoderInfo>(coeff_count_bs, s);

        // We need m powers of the primitive 2n-th root, m = 2n

        sparse_bootstrap_root_powers_.reserve(m_bs);
        sparse_bootstrap_rotation_group_.reserve(slots_half_bs);


        for (size_t i = 0; i < slots_half_bs; i++)
        {
            // Set the bit-reversed locations
            sparse_bootstrap_rotation_group_[i] = pos;

            // Next primitive root
            pos *= gen; // 5^i mod m
            pos &= (m_bs - 1);
        }

        // Powers of the primitive 2n-th root have 4-fold symmetry
        if (m_bs >= 8)
        {
            sparse_bootstrap_complex_roots_ = std::make_unique<phantom::util::ComplexRoots>(phantom::util::ComplexRoots(static_cast<size_t>(m_bs)));
            for (size_t i = 0; i < m_bs; i++)
            {
                sparse_bootstrap_root_powers_[i] = sparse_bootstrap_complex_roots_->get_root(i);
            }
        }
        else if (m_bs == 4)
        {


            sparse_bootstrap_root_powers_[0] = {1, 0};
            sparse_bootstrap_root_powers_[1] = {0, 1};
            sparse_bootstrap_root_powers_[2] = {-1, 0};
            sparse_bootstrap_root_powers_[3] = {0, -1};
        }

        cudaMemcpyAsync(sparse_bootstrap_gpu_ckks_msg_vec_->twiddle(), sparse_bootstrap_root_powers_.data(), m_bs * sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(sparse_bootstrap_gpu_ckks_msg_vec_->mul_group(), sparse_bootstrap_rotation_group_.data(), slots_half_bs * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);

        temp_parms.set_poly_modulus_degree(coeff_count_bs);
        sparse_bootstrap_context_.emplace(temp_parms); // Constructs another one
    }
};
