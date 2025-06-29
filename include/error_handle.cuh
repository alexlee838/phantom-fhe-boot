#ifndef ERROR_HANDLING_CUH
#define ERROR_HANDLING_CUH

#include <cstdio>
#include <cstdlib>
#include <string>

inline void throwError(const std::string& msg) {
    fprintf(stderr, "Error: %s\n", msg.c_str());
    exit(EXIT_FAILURE);
}

#endif  // ERROR_HANDLING_CUH
