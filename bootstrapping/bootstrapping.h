#pragma once

#include <iomanip>
#include <iostream>
#include <cmath>
#include "phantom.h"
#include "context.cuh"
#include "bootstrap.cuh"
#include <cuda_runtime.h>
#include <random>
#include "timer.h"

#define AUX_MOD 60

inline uint32_t ComputeNumLargeDigits(uint32_t numLargeDigits, uint32_t multDepth) {
    if (numLargeDigits > 0)
        return numLargeDigits;
    if (multDepth > 3)  // if more than 4 towers, use 3 digits
        return 3;
    if (multDepth > 0)  // if 2, 3 or 4 towers, use 2 digits
        return 2;
    return 1;  // if 1 tower, use one digit
}

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

double ComputeMSE(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum / v1.size();
}

double ComputeRMSE(const std::vector<double>& v1, const std::vector<double>& v2) {
    return std::sqrt(ComputeMSE(v1, v2));
}


std::vector<double> GenerateRandomVector(size_t numSlots, double min_val = 1.0, double max_val = 5.0) {
    std::vector<double> random_values(numSlots);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister engine
    std::uniform_real_distribution<double> dis(min_val, max_val);

    for (size_t i = 0; i < numSlots; ++i) {
        random_values[i] = dis(gen);
    }
    return random_values;
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

    std::cout << std::endl;
    for (std::size_t i = 0; i < coeff_modulus_size; i++) {
        std::cout << coeff_modulus[i].value() << " ,  ";
    }
    std::cout << std::endl
            << std::endl;

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
