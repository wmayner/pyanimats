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
