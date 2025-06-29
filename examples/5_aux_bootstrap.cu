#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>
#include "example.h"
#include "phantom.h"
#include "util.cuh"
#include "bootstrap.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
#define EPSINON 0.001

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs)
{
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs)
{
    return fabs(lhs - rhs) < EPSINON;
}

void example_eval_chebyshev(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS ChebyShev test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    std::vector<cuDoubleComplex> input{make_cuDoubleComplex(-3.0, 0), make_cuDoubleComplex(-2.0, 0), make_cuDoubleComplex(-1.0, 0),
                                       make_cuDoubleComplex(0.0, 0), make_cuDoubleComplex(1.0, 0), make_cuDoubleComplex(2.0, 0), make_cuDoubleComplex(3.0, 0)};

    std::vector<double> coefficients{9, -17.25, 4.5, -6.75, -0};
    std::vector<cuDoubleComplex> output1{make_cuDoubleComplex(33, 0), make_cuDoubleComplex(10, 0), make_cuDoubleComplex(1, 0),
                                         make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(-2, 0), make_cuDoubleComplex(-15, 0)};

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    cout << "Message vector: " << endl;
    print_vector(input, 3, 7);

    PhantomPlaintext x_plain;

    encoder.encode(context, input, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    x_cipher.PreComputeScale(context, scale);

    auto m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
    auto m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

    cout << "Compute ChebyshevSeries" << endl;
    double a = -3;
    double b = 3;
    auto result_cipher = EvalChebyshevSeries(context, relin_keys, x_cipher, coefficients, a, b, m_scalingFactorsReal, m_scalingFactorsRealBig);

    PhantomPlaintext result_plain = secret_key.decrypt(context, result_cipher);
    auto result = encoder.decode<cuDoubleComplex>(context, result_plain);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < input.size(); i++)
    {
        correctness &= ((fabs(output1[i].x - result[i].x) < EPSINON) && fabs(output1[i].y - result[i].y) < EPSINON);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic multiplication error");
    result.clear();
    input.clear();
}

void example_eval_chebyshev_sine(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS ChebyShev Sine test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    std::vector<cuDoubleComplex> input{make_cuDoubleComplex(-1.0, 0), make_cuDoubleComplex(-0.8, 0), make_cuDoubleComplex(-0.6, 0),
                                       make_cuDoubleComplex(-0.4, 0), make_cuDoubleComplex(-0.2, 0), make_cuDoubleComplex(0.0, 0), make_cuDoubleComplex(0.2, 0), make_cuDoubleComplex(0.4, 0), make_cuDoubleComplex(0.6, 0), make_cuDoubleComplex(0.8, 0),
                                       make_cuDoubleComplex(1.0, 0)};

    std::vector<double> coefficients{
        0., -0.0178446, 0., -0.0171187, 0., -0.0155856, 0., -0.0131009, 0., -0.00949759,
        0., -0.00465513, 0., 0.00139902, 0., 0.00836141, 0., 0.0155242, 0., 0.0217022,
        0., 0.0253027, 0., 0.0246365, 0., 0.0185273, 0., 0.00714273, 0., -0.00725482,
        0., -0.0201827, 0., -0.0260483, 0., -0.0207132, 0., -0.00473479, 0., 0.0147661,
        0., 0.0261764, 0., 0.0203168, 0., -0.00103552, 0., -0.0225101, 0., -0.0248192,
        0., -0.00315799, 0., 0.0226844, 0., 0.0238252, 0., -0.00403513, 0., -0.0276106,
        0., -0.0133143, 0., 0.0213882, 0., 0.0230787, 0., -0.0143638, 0., -0.0270401,
        0., 0.0116019, 0., 0.0278743, 0., -0.0149975, 0., -0.025194, 0., 0.0242296,
        0., 0.0143133, 0., -0.0334779, 0., 0.00994475, 0., 0.0256291, 0., -0.0359815,
        0., 0.0150778, 0., 0.0173112, 0., -0.0403029, 0., 0.0463332, 0., -0.039547,
        0., 0.0277765, 0., -0.0168089, 0., 0.00899558, 0., -0.00433006, 0., 0.00189728,
        0., -0.000763553, 0., 0.000284227, 0., -0.0000984182, 0., 0.0000318501, 0., -9.67162e-6,
        0., 2.76517e-6, 0., -7.46488e-7, 0., 1.90362e-7, 0., -4.39544e-8, 0.};

    std::vector<cuDoubleComplex> output1{make_cuDoubleComplex(6.80601e-09, 0), make_cuDoubleComplex(0.151365, 0), make_cuDoubleComplex(0.0935489, 0),
                                         make_cuDoubleComplex(-0.0935489, 0), make_cuDoubleComplex(-0.151365, 0), make_cuDoubleComplex(0.0, 0), make_cuDoubleComplex(0.151365, 0), make_cuDoubleComplex(0.0935489, 0), make_cuDoubleComplex(-0.0935489, 0), make_cuDoubleComplex(-0.151365, 0), make_cuDoubleComplex(-6.80601e-09, 0)};

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    cout << "Message vector: " << endl;
    print_vector(input, 3, 7);

    PhantomPlaintext x_plain;

    encoder.encode(context, input, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    x_cipher.PreComputeScale(context, scale);

    auto m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
    auto m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

    double a = -1;
    double b = 1;
    auto result_cipher = EvalChebyshevSeries(context, relin_keys, x_cipher, coefficients, a, b, m_scalingFactorsReal, m_scalingFactorsRealBig);

    PhantomPlaintext result_plain = secret_key.decrypt(context, result_cipher);
    auto result = encoder.decode<cuDoubleComplex>(context, result_plain);
    cout << "Result vector: " << endl;
    result.resize(input.size());
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < input.size(); i++)
    {
        correctness &= ((fabs(output1[i].x - result[i].x) < EPSINON) && fabs(output1[i].y - result[i].y) < EPSINON);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic multiplication error");
    result.clear();
    input.clear();
}

void example_ckks_cal_const(PhantomContext &context, const double &scale)
{

    std::cout << "Example: CKKS Cipher multiply const" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> msg_vec, result;
    double rand_const;

    double rand_real, rand_imag;

    size_t msg_size = slot_count;
    auto sizeQ = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    auto &parms = context.get_context_data(context.get_first_index()).parms();

    msg_vec.reserve(msg_size);

    for (size_t i = 0; i < msg_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        msg_vec.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    rand_const = (double)rand() / RAND_MAX;

    std::cout << "Constant ";
    std::cout << rand_const << std::endl;

    PhantomPlaintext plain;

    encoder.encode(context, msg_vec, scale, plain);

    PhantomCiphertext cipher;
    public_key.encrypt_asymmetric(context, plain, cipher);

    cipher.PreComputeScale(context, scale);
    auto m_scalingFactorsReal = cipher.getScalingFactorsReal();
    auto m_scalingFactorsRealBig = cipher.getScalingFactorsRealBig();

    EvalMultConstInplace(context, cipher, rand_const, m_scalingFactorsReal);

    secret_key.decrypt(context, cipher, plain);
    encoder.decode(context, plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    verifyResults(result, msg_vec, rand_const, msg_size, Ops::MUL);

    std::cout << "Example: CKKS Cipher Add Const" << std::endl;

    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    rand_const = (double)rand() / RAND_MAX;

    std::cout << "Constant ";
    std::cout << rand_const << std::endl;

    encoder.encode(context, msg_vec, scale, plain);
    public_key.encrypt_asymmetric(context, plain, cipher);
    EvalAddConstInPlaceWrap(context, cipher, rand_const, m_scalingFactorsReal, m_scalingFactorsRealBig);
    secret_key.decrypt(context, cipher, plain);
    encoder.decode(context, plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    verifyResults(result, msg_vec, rand_const, msg_size, Ops::ADD);

    std::cout << "Example: CKKS Cipher Sub Const" << std::endl;

    cout << "Message vector: " << endl;
    print_vector(msg_vec, 3, 7);

    rand_const = (double)rand() / RAND_MAX;

    std::cout << "Constant ";
    std::cout << rand_const << std::endl;

    encoder.encode(context, msg_vec, scale, plain);
    public_key.encrypt_asymmetric(context, plain, cipher);
    EvalSubConstInPlace(context, cipher, rand_const, m_scalingFactorsReal, m_scalingFactorsRealBig);
    secret_key.decrypt(context, cipher, plain);
    encoder.decode(context, plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    verifyResults(result, msg_vec, rand_const, msg_size, Ops::SUB);

    msg_vec.clear();
    result.clear();
}

void example_ckks_add_sub_auto(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Add Sub test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    size_t y_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    y_msg.reserve(y_size);
    for (size_t i = 0; i < y_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    x_cipher.PreComputeScale(context, scale);
    auto m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
    auto m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

    cout << "Compute x +- y." << endl;
    PhantomCiphertext cipher_plus = EvalAddAuto(context, x_cipher, y_cipher, m_scalingFactorsReal, m_scalingFactorsRealBig);
    PhantomCiphertext cipher_minus = EvalSubAuto(context, x_cipher, y_cipher, m_scalingFactorsReal, m_scalingFactorsRealBig);

    PhantomPlaintext plain_plus = secret_key.decrypt(context, cipher_plus);
    PhantomPlaintext plain_minus = secret_key.decrypt(context, cipher_minus);
    auto result_plus = encoder.decode<cuDoubleComplex>(context, plain_plus);
    auto result_minus = encoder.decode<cuDoubleComplex>(context, plain_minus);

    cout << "Result vector: " << endl;
    print_vector(result_plus, 3, 7);
    print_vector(result_minus, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_plus[i] == cuCadd(x_msg[i], y_msg[i]);
        correctness &= result_minus[i] == cuCsub(x_msg[i], y_msg[i]);
    }

    if (!correctness)
        throw std::logic_error("Homomorphic Add Minus error");

    result_plus.clear();
    result_minus.clear();
    x_msg.clear();
    y_msg.clear();
}

void example_ckks_mul_auto(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS HomMul test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    size_t y_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    y_msg.reserve(y_size);
    for (size_t i = 0; i < y_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext y_plain;
    // PhantomPlaintext xy_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher;
    PhantomCiphertext y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);
    mod_switch_to_next_inplace(context, x_cipher);

    x_cipher.PreComputeScale(context, scale);
    auto m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
    auto m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

    cout << "Compute x^2*y^2." << endl;
    PhantomCiphertext xy_cipher = EvalMultAuto(context, x_cipher, y_cipher, relin_keys, m_scalingFactorsReal, m_scalingFactorsRealBig);
    PhantomCiphertext x2y_cipher = EvalMultAuto(context, xy_cipher, x_cipher, relin_keys, m_scalingFactorsReal, m_scalingFactorsRealBig);
    PhantomCiphertext x2y2_cipher = EvalMultAuto(context, x2y_cipher, y_cipher, relin_keys, m_scalingFactorsReal, m_scalingFactorsRealBig);
    PhantomCiphertext x2y2_cipher_with_sq = EvalSquare(context, xy_cipher, relin_keys, m_scalingFactorsReal, m_scalingFactorsRealBig);
    PhantomCiphertext xy_cipher_ = x_cipher;
    EvalMultAutoInplace(context, xy_cipher_, y_plain, m_scalingFactorsReal, m_scalingFactorsRealBig);

    PhantomPlaintext x2y2_plain = secret_key.decrypt(context, x2y2_cipher);
    PhantomPlaintext x2y2_plain_sq = secret_key.decrypt(context, x2y2_cipher_with_sq);
    PhantomPlaintext xy_plain = secret_key.decrypt(context, xy_cipher_);
    auto result = encoder.decode<cuDoubleComplex>(context, x2y2_plain);
    auto result_sq = encoder.decode<cuDoubleComplex>(context, x2y2_plain_sq);
    auto result_pl = encoder.decode<cuDoubleComplex>(context, xy_plain);

    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);
    print_vector(result_sq, 3, 7);
    print_vector(result_pl, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result[i] == cuCmul(cuCmul(x_msg[i], y_msg[i]), cuCmul(x_msg[i], y_msg[i]));
        correctness &= result_sq[i] == cuCmul(cuCmul(x_msg[i], y_msg[i]), cuCmul(x_msg[i], y_msg[i]));
        correctness &= result_pl[i] == cuCmul(x_msg[i], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic multiplication error");
    result.clear();
    result_sq.clear();
    x_msg.clear();
    y_msg.clear();
}

void example_ckks_hoisting(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Hoisting test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    // PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto n = key_parms.poly_modulus_degree();

    std::vector<int32_t> rot_idx = {1, 8, 32, 64, 256, 512, 768, 1024, 1280, 1536, 1600, 1664, 1728, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 1992, 2000, 2008, 2016, 2020, 2024, 2028, 2032, 2036, 2040,
                                    2041, 2042, 2043, 2044, 2045, 2046, 2047};

    std::vector<PhantomCiphertext> rotated_ciphers(rot_idx.size());
    std::vector<PhantomPlaintext> rotated_plaintexts(rot_idx.size());
    std::vector<std::vector<cuDoubleComplex>> rotated_msgs(rot_idx.size());

    PhantomGaloisKeyFused fused_keys = secret_key.EvalRotateKeyGen(context, rot_idx);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);

    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext x_rot_plain;

    encoder.encode(context, x_msg, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    mod_switch_to_next_inplace(context, x_cipher);

    cout << "Compute, rot vector x. (Fused Version)" << endl;
    // rotate_inplace(context, x_cipher, step, galois_keys);
    auto digits = EvalFastRotationPrecompute(context, x_cipher);

    for (int i = 0; i < rot_idx.size(); i++)
    {

        rotated_ciphers[i] = EvalFastRotationExt(context, x_cipher, fused_keys, rot_idx[i], digits, true);
        rotated_ciphers[i] = KeySwitchDown(context, rotated_ciphers[i]);
        secret_key.decrypt(context, rotated_ciphers[i], rotated_plaintexts[i]);
        encoder.decode(context, rotated_plaintexts[i], rotated_msgs[i]);
    }

    print_vector(rotated_msgs[0], 3, 7);

    bool correctness = true;
    for (size_t j = 0; j < rot_idx.size(); j++)
    {
        for (size_t i = 0; i < x_size; i++)
        {
            correctness &= rotated_msgs[j][i] == x_msg[(i + rot_idx[j]) % x_size];
        }
        if (!correctness)
        {
            std::cout << j << std::endl;
            throw std::logic_error("Homomorphic rotation error");
        }
    }

    x_msg.clear();
    for (int i = 0; i < rot_idx.size(); i++)
    {
        rotated_msgs[i].clear();
    }
}

void example_ext(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Extended Arith test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    int32_t rot_idx = 5;
    PhantomGaloisKeyFused galois_keys = secret_key.EvalRotateKeyGen(context, {rot_idx});

    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    y_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain, y_plain, y_plain_ext;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);
    encoder.encode_ext(context, y_msg, scale, y_plain_ext, 2);

    PhantomCiphertext x_cipher, y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    size_t size_Ql = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool().base_Ql().size();

    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    size_t size_P = context.get_context_data(0).parms().special_modulus_size();

    mod_switch_to_next_inplace(context, x_cipher);
    mod_switch_to_next_inplace(context, y_cipher);

    // std::cout << x_cipher.chain_index() << std::endl;

    PhantomCiphertext x_cipher_ext = KeySwitchExt(context, x_cipher);

    auto digits = EvalFastRotationPrecompute(context, x_cipher);
    auto x_rot_ext = EvalFastRotationExt(context, x_cipher, galois_keys, rot_idx, digits, true);

    PhantomCiphertext y_cipher_ext = KeySwitchExt(context, y_cipher);
    PhantomCiphertext x_plus_y_ext = EvalAddExt(context, x_cipher_ext, y_cipher_ext);
    PhantomCiphertext x_mul_y_ext = EvalMultExt(context, x_cipher_ext, y_plain_ext);
    PhantomCiphertext x_rot_add_y_ext = EvalAddExt(context, x_rot_ext, y_cipher_ext);
    PhantomCiphertext x_rot_mul_y_ext = EvalMultExt(context, x_rot_ext, y_plain_ext);

    PhantomCiphertext result_cipher_rot = KeySwitchDown(context, x_rot_ext);
    PhantomCiphertext result_cipher_plus = KeySwitchDown(context, x_plus_y_ext);
    PhantomCiphertext result_cipher_mul = KeySwitchDown(context, x_mul_y_ext);
    PhantomCiphertext result_cipher_rot_add = KeySwitchDown(context, x_rot_add_y_ext);
    PhantomCiphertext result_cipher_rot_mul = KeySwitchDown(context, x_rot_mul_y_ext);

    PhantomPlaintext result_plain_rot = secret_key.decrypt(context, result_cipher_rot);
    PhantomPlaintext result_plain_plus = secret_key.decrypt(context, result_cipher_plus);
    PhantomPlaintext result_plain_mul = secret_key.decrypt(context, result_cipher_mul);
    PhantomPlaintext result_plain_rot_add = secret_key.decrypt(context, result_cipher_rot_add);
    PhantomPlaintext result_plain_rot_mul = secret_key.decrypt(context, result_cipher_rot_mul);

    auto result_rot = encoder.decode<cuDoubleComplex>(context, result_plain_rot);
    auto result_plus = encoder.decode<cuDoubleComplex>(context, result_plain_plus);
    auto result_mul = encoder.decode<cuDoubleComplex>(context, result_plain_mul);
    auto result_add_rot = encoder.decode<cuDoubleComplex>(context, result_plain_rot_add);
    auto result_mul_rot = encoder.decode<cuDoubleComplex>(context, result_plain_rot_mul);

    cout << "Rot Vec: " << endl;
    print_vector(result_rot, 3, 7);

    cout << "Extend Add" << endl;
    print_vector(result_plus, 3, 7);

    cout << "Extend Mul" << endl;
    print_vector(result_mul, 3, 7);

    cout << "Extend Rot Add" << endl;
    print_vector(result_add_rot, 3, 7);

    cout << "Extend Rot Mul" << endl;
    print_vector(result_mul_rot, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_rot[i] == x_msg[(i + rot_idx) % x_size];
    }
    if (!correctness)
        throw std::logic_error("Homomorphic rotation error");

    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_plus[i] == cuCadd(x_msg[i], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic extension addition error");

    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_mul[i] == cuCmul(x_msg[i], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic extension multiplication error");

    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_add_rot[i] == cuCadd(x_msg[(i + rot_idx) % x_size], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic extension rotation addition error");

    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result_mul_rot[i] == cuCmul(x_msg[(i + rot_idx) % x_size], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic extension rotation multiplication error");

    result_add_rot.clear();
    result_mul_rot.clear();
    result_plus.clear();
    result_mul.clear();
    x_msg.clear();
    y_msg.clear();
}

void example_rot_lazy_add(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Extended Arith Test (w/ Rotation)" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // const auto &s = cudaStreamPerThread;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    y_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        y_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);
    print_vector(y_msg, 3, 7);

    PhantomPlaintext x_plain, y_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    encoder.encode(context, y_msg, scale, y_plain);

    PhantomCiphertext x_cipher, y_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);
    auto &rns_tool = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool();
    size_t size_Ql = context.get_context_data(x_cipher.chain_index()).gpu_rns_tool().base_Ql().size();
    size_t size_Q = context.get_context_data(context.get_first_index()).parms().coeff_modulus().size();
    size_t size_P = context.get_context_data(0).parms().special_modulus_size();
    auto modulus_QP = context.gpu_rns_tables().modulus();
    auto poly_degree = x_cipher.poly_modulus_degree();

    int32_t rot_idx = 5;
    auto x_rot = rotate_c0(context, x_cipher, rot_idx);
    PhantomCiphertext y_cipher_ext = KeySwitchExt(context, y_cipher);

    PhantomGaloisKeyFused galois_keys = secret_key.EvalRotateKeyGen(context, {rot_idx});

    auto x_digits = EvalFastRotationPrecompute(context, x_cipher);
    x_cipher = EvalAddExt(context, EvalFastRotationExt(context, x_cipher, galois_keys, rot_idx, x_digits, false), y_cipher_ext);

    PhantomCiphertext result_cipher = KeySwitchDown(context, x_cipher);

    auto result_coeff_mod = result_cipher.coeff_modulus_size();

    auto result_poly_deg = result_cipher.poly_modulus_degree();
    PhantomCiphertext result_cipher_save = result_cipher;

    add_two_poly_inplace(context, result_cipher.data(), x_rot.get(), x_cipher.chain_index());

    PhantomPlaintext result_plain = secret_key.decrypt(context, result_cipher);

    auto result = encoder.decode<cuDoubleComplex>(context, result_plain);

    cout << "Result vector: " << endl;

    cout << "Extend Rot Add" << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result[i] == cuCadd(x_msg[(i + rot_idx) % x_size], y_msg[i]);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic Rotation and Addition error");

    x_msg.clear();
    y_msg.clear();
    result.clear();
}

void example_mult_mono(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Monomial Mult" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // const auto &s = cudaStreamPerThread;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, y_msg;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }

    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain;

    encoder.encode(context, x_msg, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    auto poly_degree = x_cipher.poly_modulus_degree();
    size_t M = 2 * poly_degree;

    MultByMonomialInPlace(context, x_cipher, 3 * M / 4);
    MultByMonomialInPlace(context, x_cipher, M / 4);

    PhantomPlaintext result_plain = secret_key.decrypt(context, x_cipher);

    auto result = encoder.decode<cuDoubleComplex>(context, result_plain);

    cout << "Result vector: " << endl;

    cout << "Mult with Monomial" << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result[i] == x_msg[i];
    }
    if (!correctness)
        throw std::logic_error("Homomorphic Mult with Monomial error");

    x_msg.clear();
    result.clear();
}

void example_ckks_enc_sparse(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Encrypt/Decrpyt sparse vector" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    size_t val_size = slot_count / 4;
    cout << "Number of values: " << val_size << endl;

    vector<cuDoubleComplex> input(val_size);
    double rand_real;
    double rand_imag;
    // srand(time(0));
    for (size_t i = 0; i < val_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = 0; //(double)rand() / RAND_MAX;
        input[i] = make_cuDoubleComplex(rand_real, rand_imag);
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    PhantomPlaintext x_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.set_sparse_encode(context.get_context_data(0).parms(), val_size * 2);
    encoder.encode_sparse(context, input, scale, x_plain, 1);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    PhantomPlaintext result_plain = secret_key.decrypt(context, x_cipher);

    bool correctness = true;

    // Decode check
    vector<cuDoubleComplex> result;
    encoder.decode(context, result_plain, result);
    // encoder.decode_sparse(context, result_plain, val_size, result);

    print_vector(result, 3, 7);
    for (size_t i = 0; i < val_size; i++)
    {
        correctness &= result[i] == input[i];
    }
    if (!correctness)
        throw std::logic_error("encode/decode complex vector error");
    result.clear();
}

void example_ckks_enc_sparse_ext(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS Encrypt/Decrpyt sparse Ext vector" << std::endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    size_t val_size = slot_count / 4;
    cout << "Number of values: " << val_size << endl;

    vector<cuDoubleComplex> input(val_size);
    double rand_real;
    double rand_imag;
    // srand(time(0));
    for (size_t i = 0; i < val_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = 0; //(double)rand() / RAND_MAX;
        input[i] = make_cuDoubleComplex(rand_real, rand_imag);
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    PhantomPlaintext x_plain, x_plain_ext;
    PhantomCiphertext x_cipher;

    print_line(__LINE__);
    cout << "Encode input vectors." << endl;

    encoder.set_sparse_encode(context.get_context_data(0).parms(), val_size * 2);
    encoder.encode_sparse_ext(context, input, scale, x_plain_ext, 1);
    encoder.encode_sparse(context, input, scale, x_plain, 1);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    PhantomCiphertext x_cipher_ext = KeySwitchExt(context, x_cipher);
    PhantomCiphertext x_mul_x_ext = EvalMultExt(context, x_cipher_ext, x_plain_ext);
    PhantomCiphertext result_cipher = KeySwitchDown(context, x_mul_x_ext);

    PhantomPlaintext result_plain = secret_key.decrypt(context, result_cipher);

    bool correctness = true;

    // Decode check
    vector<cuDoubleComplex> result;
    encoder.decode_sparse(context, result_plain, val_size, result);

    print_vector(result, 3, 7);
    for (size_t i = 0; i < val_size; i++)
    {
        correctness &= result[i] == cuCmul(input[i], input[i]);
    }
    if (!correctness)
        throw std::logic_error("encode/decode complex vector error");
    result.clear();
}

void example_ckks_rotation_sparse(PhantomContext &context, const double &scale)
{
    std::cout << "Example: CKKS HomRot test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    // PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto n = key_parms.poly_modulus_degree();

    PhantomGaloisKeyFused fused_keys = secret_key.EvalRotateKeyGen(context, {1, 11, 15, -33});

    int step = -33;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, result;
    double rand_real, rand_imag;

    size_t x_size = slot_count / 8;
    encoder.set_sparse_encode(context.get_context_data(0).parms(), x_size * 2);

    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext x_rot_plain;

    encoder.encode_sparse(context, x_msg, scale, x_plain, 1);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, rot vector x. (Fused Version)" << endl;
    // rotate_inplace(context, x_cipher, step, galois_keys);
    PhantomCiphertext x_cipher_rot;
    EvalRotateFused(context, fused_keys, x_cipher, x_cipher_rot, step);

    // secret_key.decrypt(context, x_cipher, x_rot_plain);
    secret_key.decrypt(context, x_cipher_rot, x_rot_plain);

    encoder.decode(context, x_rot_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result[i] == x_msg[(i + step + x_size) % x_size];
    }
    if (!correctness)
        throw std::logic_error("Homomorphic rotation error");
    result.clear();
    x_msg.clear();

    std::cout << "Example: CKKS HomConj test" << std::endl;

    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++)
    {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_conj_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, conjugate vector x. (Fused)" << endl;
    // rotate_inplace(context, x_cipher, 0, galois_keys);
    PhantomCiphertext x_cipher_conj;
    EvalConjFused(context, fused_keys, x_cipher, x_cipher_conj);

    secret_key.decrypt(context, x_cipher_conj, x_conj_plain);

    encoder.decode(context, x_conj_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    correctness = true;
    for (size_t i = 0; i < x_size; i++)
    {
        correctness &= result[i] == make_cuDoubleComplex(x_msg[i].x, -x_msg[i].y);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic conjugate error");
    result.clear();
    x_msg.clear();
}
void example_aux_bootstrap()
{
    int device_id = 2; // Change to the desired GPU index
    cudaSetDevice(device_id);

    print_example_banner("Example: Boot Strapping Helper Functions");

    srand(time(NULL));

    std::vector v_alpha = {1, 2, 3, 4};
    for (auto alpha : v_alpha)
    {
        EncryptionParameters parms(scheme_type::ckks);

        size_t poly_modulus_degree = 1 << 15;
        double scale = pow(2.0, 59);
        switch (alpha)
        {
        case 1:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(
                CoeffModulus::Create(poly_modulus_degree, {60, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                                                           59, 59, 59, 59, 59, 59, 59, 59, 59, 60}));
            break;
        case 2:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60}));
            parms.set_special_modulus_size(alpha);
            break;
        case 3:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60}));
            parms.set_special_modulus_size(alpha);
            break;
        case 4:
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree, {60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60}));
            // hybrid key-switching
            parms.set_special_modulus_size(alpha);
            break;
        case 15:
            poly_modulus_degree = 1 << 16;
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::Create(
                poly_modulus_degree,
                {60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
            parms.set_special_modulus_size(alpha);
            scale = pow(2.0, 59);
            break;
        default:
            throw std::invalid_argument("unsupported alpha params");
        }

        PhantomContext context(parms);

        print_parameters(context);
        cout << endl;

        // example_ckks_cal_const(context, scale);
        // example_ckks_mul_auto(context, scale);
        // example_ckks_enc_sparse(context, scale);
        example_ckks_rotation_sparse(context, scale);
        // example_ckks_add_sub_auto(context, scale);
        // example_eval_chebyshev(context, scale);
        // example_eval_chebyshev_sine(context, scale);
        // example_ckks_hoisting(context, scale);
        // example_ext(context, scale);
        // example_rot_lazy_add(context, scale);
        // example_mult_mono(context, scale);
    }
}