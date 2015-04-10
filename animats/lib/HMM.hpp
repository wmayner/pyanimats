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

class HMMU{
 public:
    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;
    vector<unsigned char> ins, outs;
    unsigned char numInputs, numOutputs;

    HMMU(vector<unsigned char> &genome, int start);
    void update(unsigned char *currentStates, unsigned char *nextStates);
    ~HMMU();
};

#endif  // SRC_HMM_H_
