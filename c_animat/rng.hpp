// rng.hpp

#ifndef ANIMAT_RNG_H_
#define ANIMAT_RNG_H_

#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>


static std::mt19937 mersenne(1729);
static std::uniform_int_distribution<int> uniform_int_dist(0, RAND_MAX);
static std::uniform_real_distribution<double> uniform_double_dist(0, 1);
static std::uniform_int_distribution<int> uniform_char_int_dist(0, 255);

double randDouble();
int randInt();
int randCharInt();

void seedRNG(int s);

std::string getState();
void setState(std::string state);


#endif  // ANIMAT_RNG_H_
