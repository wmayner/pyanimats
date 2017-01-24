// rng.hpp

#include "./rng.hpp"


void seedRNG(int s) {
    mersenne.seed(s);
}

int randInt() {
    return uniform_int_dist(mersenne);
}

double randDouble() {
    return uniform_double_dist(mersenne);
}

int randCharInt() {
    return uniform_char_int_dist(mersenne);
}

int randBitInt() {
    // Return a random integer such that the bits are also uniformly
    // distributed
    return uniform_int_dist_pow2(mersenne);
}

std::string getState() {
    std::stringstream stream;
    stream << mersenne;
    return stream.str();
}

void setState(std::string state) {
    std::stringstream stream;
    stream << state;
    stream >> mersenne;
}
