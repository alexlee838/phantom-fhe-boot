/*

Example for CKKS bootstrapping

*/

#include "bootstrapping.h"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

void SimpleBootstrapExample();
void SparseBootStrapping();


double compute_bit_precision(const std::vector<double>& ref, const std::vector<double>& actual) {
    if (ref.size() != actual.size()) {
        std::cerr << "Size mismatch in compute_bit_precision!\n";
        return 0.0;
    }

    double sum_bit_precision = 0.0;
    int valid_count = 0;

    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i];
        double a = actual[i];

        if (std::abs(r) < 1e-20) continue; // skip near-zero reference

        double rel_error = std::abs(r - a) / std::abs(r);
        if (rel_error < 1e-40) rel_error = 1e-40; // avoid log2(0)

        double bits = -std::log2(rel_error);
        sum_bit_precision += bits;
        ++valid_count;
    }

    return (valid_count == 0) ? 0.0 : sum_bit_precision / valid_count;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s [simple|sparse]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "simple") == 0)
    {
        SimpleBootstrapExample();
    }
    else if (strcmp(argv[1], "sparse") == 0)
    {
        SparseBootStrapping();
    }
    else
    {
        printf("Invalid argument: %s\n", argv[1]);
        printf("Usage: %s [simple|sparse]\n", argv[0]);
        return 1;
    }

    return 0;
}

void SimpleBootstrapExample()
{
    {
        EncryptionParameters parameters(scheme_type::ckks);

        size_t N = 1 << 16;

        int device_id = 0; // Change to the desired GPU index
        cudaSetDevice(device_id);

        uint32_t dcrtBits = 59;
        uint32_t firstMod = 60;
        uint32_t levelsAvailableAfterBootstrap = 11;
        std::vector<uint32_t> levelBudget = { 2, 2 };

        uint32_t depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget);
        std::cout << "Bootstrap depth : " << depth + 1 << std::endl;
        uint32_t numLargeDigits = ComputeNumLargeDigits(0, depth);
        auto special_modulus_size = 10; // static_cast<uint32_t>(std::ceil(((firstMod + dcrtBits * (numLargeDigits - 1))) / AUX_MOD));

        std::vector<int> mod_vec = {};

        for (int i = 0; i < depth + 1 + special_modulus_size; i++)
        {
            if (i == 0)
            {
                mod_vec.push_back(firstMod);
            }

            else
            {
                if (i < depth + 1)
                {
                    mod_vec.push_back(dcrtBits);
                }
                else
                {
                    mod_vec.push_back(AUX_MOD);
                }
            }
        }
        std::cout << std::endl;
        std::cout << "Mod Size : " << mod_vec.size() << std::endl;

        parameters.set_poly_modulus_degree(N);
        parameters.set_special_modulus_size(special_modulus_size);
        parameters.set_coeff_modulus(CoeffModulus::Create(N, mod_vec));
        double scale = pow(2.0, 59);

        uint32_t numSlots = N / 2;
        std::cout << "CKKS scheme is using ring dimension " << N << std::endl
            << std::endl;

        Timer::startGPUTimer("Context Creation");
        PhantomContext context(parameters);
        Timer::stopGPUTimer("Context Creation");

        PhantomSecretKey secret_key(context);
        PhantomPublicKey public_key = secret_key.gen_publickey(context);

        PhantomCKKSEncoder encoder(context);

        std::vector<double> x = GenerateRandomVector(numSlots);

        size_t encodedLength = x.size();

        PhantomPlaintext x_plain;
        PhantomCiphertext x_cipher;

        Timer::startGPUTimer("Encoding");
        encoder.encode(context, x, scale, x_plain);
        Timer::stopGPUTimer("Encoding");

        Timer::startGPUTimer("Encryption");
        public_key.encrypt_asymmetric(context, x_plain, x_cipher);
        Timer::stopGPUTimer("Encryption");

        x_cipher.PreComputeScale(context, scale);
        std::vector<double> m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
        std::vector<double> m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

        for (int i = 0; i < 25; i++)
        {
            EvalMultConstInplace(context, x_cipher, 1, m_scalingFactorsReal);
        }

        FHECKKSRNS bootstrapper(encoder);
        std::cout << "before setup"  << std::endl;

        Timer::startGPUTimer("Bootstrap Setup");
        bootstrapper.EvalBootstrapSetup(context, levelBudget, scale, m_scalingFactorsReal, m_scalingFactorsRealBig);
        Timer::stopGPUTimer("Bootstrap Setup");


        std::cout << "setup done"  << std::endl;
        Timer::startGPUTimer("Multiplication KeyGen");
        bootstrapper.EvalMultKeyGen(secret_key, context);
        Timer::stopGPUTimer("Multiplication KeyGen");

        Timer::startGPUTimer("Bootstrap KeyGen");
        bootstrapper.EvalBootstrapKeyGen(secret_key, context, numSlots);
        Timer::stopGPUTimer("Bootstrap KeyGen");

        std::cout << "Message vector: " << std::endl;
        print_vector(x, 3, 7);

        std::cout << "Before Bootstrapping : " << mod_vec.size() - x_cipher.chain_index() - special_modulus_size - 1 << std::endl;

        PhantomCiphertext result_cipher;

        Timer::startGPUTimer("Bootstrapping");
        result_cipher = bootstrapper.EvalBootstrap(x_cipher, context);
        Timer::stopGPUTimer("Bootstrapping");

        PhantomPlaintext result_plain;
        Timer::startGPUTimer("Decryption");
        result_plain = secret_key.decrypt(context, result_cipher);
        Timer::stopGPUTimer("Decryption");

        std::vector<double> result;
        encoder.decode(context, result_plain, result);
        result.resize(x.size());
        std::cout << "Result vector: " << std::endl;
        print_vector(result, 3, 7);
        std::cout << "After Bootstrapping : " << mod_vec.size() - result_cipher.chain_index() - special_modulus_size - 1 << std::endl;
        double avg_bits = compute_bit_precision(x, result);
        std::cout << "avg : " << avg_bits << std::endl;
        Timer::printAccumulatedTimes();
    }
}

void SparseBootStrapping()
{
    // Doesn't work yet
    EncryptionParameters parameters(scheme_type::ckks);

    size_t N = 1 << 17;

    int device_id = 2; // Change to the desired GPU index
    cudaSetDevice(device_id);

    uint32_t dcrtBits = 59;
    uint32_t firstMod = 60;
    uint32_t levelsAvailableAfterBootstrap = 10;
    std::vector<uint32_t> levelBudget = { 4, 4 };

    uint32_t depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget);
    std::cout << "Bootstrap depth : " << depth << std::endl;
    uint32_t numLargeDigits = ComputeNumLargeDigits(0, depth);
    auto special_modulus_size = 11; // static_cast<uint32_t>(std::ceil(((firstMod + dcrtBits * (numLargeDigits - 1))) / AUX_MOD));

    std::vector<int> mod_vec = {};

    for (int i = 0; i < depth + 1 + special_modulus_size; i++)
    {
        if (i == 0)
        {
            mod_vec.push_back(firstMod);
        }

        else
        {
            if (i < depth + 1)
            {
                mod_vec.push_back(dcrtBits);
            }
            else
            {
                mod_vec.push_back(AUX_MOD);
            }
        }
    }
    std::cout << std::endl;
    std::cout << "Mod Size : " << mod_vec.size() << std::endl;

    parameters.set_poly_modulus_degree(N);
    parameters.set_special_modulus_size(special_modulus_size);
    parameters.set_coeff_modulus(CoeffModulus::Create(N, mod_vec));
    double scale = pow(2.0, 59);

    uint32_t numSlots = N / 8;
    std::cout << "CKKS scheme is using ring dimension " << N << std::endl
        << std::endl;

    Timer::startGPUTimer("Context Creation");
    PhantomContext context(parameters);
    Timer::stopGPUTimer("Context Creation");

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    PhantomCKKSEncoder encoder(context);

    std::vector<double> x = GenerateRandomVector(numSlots);

    size_t encodedLength = x.size();

    PhantomPlaintext x_plain;
    PhantomCiphertext x_cipher;

    Timer::startGPUTimer("Encoding");
    encoder.set_sparse_encode(context.get_context_data(0).parms(), numSlots * 2);
    encoder.encode_sparse(context, x, scale, x_plain, 1);
    Timer::stopGPUTimer("Encoding");

    Timer::startGPUTimer("Encryption");
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    Timer::stopGPUTimer("Encryption");

    x_cipher.PreComputeScale(context, scale);
    std::vector<double> m_scalingFactorsReal = x_cipher.getScalingFactorsReal();
    std::vector<double> m_scalingFactorsRealBig = x_cipher.getScalingFactorsRealBig();

    for (int i = 0; i < 30; i++)
    {
        EvalMultConstInplace(context, x_cipher, 1, m_scalingFactorsReal);
    }

    FHECKKSRNS bootstrapper(encoder);

    Timer::startGPUTimer("Bootstrap Setup");
    bootstrapper.EvalBootstrapSetup(context, levelBudget, scale, m_scalingFactorsReal, m_scalingFactorsRealBig, { 0, 0 }, numSlots);
    Timer::stopGPUTimer("Bootstrap Setup");

    Timer::startGPUTimer("Multiplication KeyGen");
    bootstrapper.EvalMultKeyGen(secret_key, context);
    Timer::stopGPUTimer("Multiplication KeyGen");

    Timer::startGPUTimer("Bootstrap KeyGen");
    bootstrapper.EvalBootstrapKeyGen(secret_key, context, numSlots);
    Timer::stopGPUTimer("Bootstrap KeyGen");

    std::cout << "Message vector: " << std::endl;
    print_vector(x, 3, 7);

    std::cout << "Before Bootstrapping : " << mod_vec.size() - x_cipher.chain_index() - special_modulus_size - 1 << std::endl;

    PhantomCiphertext result_cipher;

    Timer::startGPUTimer("Bootstrapping");
    result_cipher = bootstrapper.EvalBootstrap(x_cipher, context, numSlots);
    Timer::stopGPUTimer("Bootstrapping");

    PhantomPlaintext result_plain;
    Timer::startGPUTimer("Decryption");
    result_plain = secret_key.decrypt(context, result_cipher);
    Timer::stopGPUTimer("Decryption");

    std::vector<double> result;

    encoder.decode(context, result_plain, result);
    result.resize(x.size());
    std::cout << "Result vector: " << std::endl;
    print_vector(result, 3, 7);
    std::cout << "After Bootstrapping : " << mod_vec.size() - result_cipher.chain_index() - special_modulus_size - 1 << std::endl;
    std::cout << "MSE : " << ComputeMSE(x, result) << std::endl;
    Timer::printAccumulatedTimes();
}
