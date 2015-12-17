// HMM.h

#ifndef SRC_HMM_H_
#define SRC_HMM_H_

#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <deque>
#include <iostream>

#include "./constants.hpp"

using std::vector;

class HMM {
 public:
    HMM(vector<unsigned char> &genome, int start);
    ~HMM();

    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;
    vector<unsigned char> ins, outs;
    unsigned char numInputs, numOutputs;

    void update(unsigned char *currentStates, unsigned char *nextStates);
};

#endif  // SRC_HMM_H_
