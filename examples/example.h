#pragma once

#include <iomanip>
#include <iostream>
#include "phantom.h"
#include "context.cuh"


enum class Ops {
    ADD,
    SUB,
    MUL
};

/*
Helper function: Prints the name of the example in a fancy banner.
*/
inline void print_example_banner(std::string title) {
    if (!title.empty()) {
        std::size_t title_length = title.length();
        std::size_t banner_length = title_length + 2 * 10;
        std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
        std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

        std::cout << std::endl
                << banner_top << std::endl
                << banner_middle << std::endl
                << banner_top << std::endl;
    }
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const PhantomContext &context) {
    auto &context_data = context.get_context_data(0);
    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme()) {
        case phantom::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case phantom::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        case phantom::scheme_type::bgv:
            scheme_name = "BGV";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++) {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    // std::cout << std::endl;
    // for (std::size_t i = 0; i < coeff_modulus_size; i++) {
    //     std::cout << coeff_modulus[i].value() << " ,  ";
    // }
    // std::cout << std::endl
    //         << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == phantom::scheme_type::bfv) {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

template<typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i] << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

/*
Helper function: Prints a vector of floating-point values.
*/
inline void print_vector(std::vector<cuDoubleComplex> vec, std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

/*
Helper function: Print line number.
*/
inline void print_line(int line_number) {
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}


inline bool almostEqual(double a, double b, double epsilon = 1e-9) {
    return fabs(a - b) < epsilon;
}

inline void verifyResults(std::vector<cuDoubleComplex>& result,  std::vector<cuDoubleComplex>& msg_vec, double rand_const, size_t msg_size, Ops option) {
    
    cuDoubleComplex expected;

    for (size_t i = 0; i < msg_size; i++) {
        if(option == Ops::MUL) {
            expected = make_cuDoubleComplex(msg_vec[i].x * rand_const, msg_vec[i].y * rand_const);
        }

        else if(option == Ops::ADD) {
            expected = make_cuDoubleComplex(msg_vec[i].x + rand_const, msg_vec[i].y);
        }

        else if (option == Ops::SUB) {
            expected = make_cuDoubleComplex(msg_vec[i].x - rand_const, msg_vec[i].y);
        }

        else {
            throwError("Invalid Option");
        }


        bool realPartCorrect = almostEqual(result[i].x, expected.x);
        bool imagPartCorrect = almostEqual(result[i].y, expected.y);

        if (!realPartCorrect || !imagPartCorrect) {
            std::cout << "Mismatch found at index " << i << ":\n";
            std::cout << "Result: " << result[i].x << " + I * " << result[i].y << std::endl;
            std::cout << "Expected: " << expected.x << " + I * " << expected.y << std::endl;
            throw std::logic_error("Const Calculation Error: Numerical mismatch detected");
        }
    }

    // std::cout << "All values match within the tolerance!" << std::endl;
}


void example_bfv_basics();

void example_bfv_batch_unbatch();

void example_bfv_encrypt_decrypt();

void example_bfv_encrypt_decrypt_asym();

void example_bfv_add();

void example_bfv_sub();

void example_bfv_mul();

void example_bfv_square();

void example_bfv_add_plain();

void example_bfv_sub_plain();

void example_bfv_mul_many_plain();

void example_bfv_mul_one_plain();

void example_bfv_rotate_column();

void example_bfv_rotate_row();

void example_encoders();

void examples_bgv();

void examples_ckks();

void example_bfv_encrypt_decrypt_hps();

void example_bfv_encrypt_decrypt_hps_asym();

void example_bfv_hybrid_key_switching();

void example_bfv_multiply_correctness();

void example_bfv_multiply_benchmark();

void example_kernel_fusing();

void example_aux_bootstrap();
