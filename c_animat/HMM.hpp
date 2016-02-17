// HMM.h

#ifndef SRC_HMM_H_
#define SRC_HMM_H_

#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <deque>
#include <iostream>

using std::vector;

class HMM {
 public:
    HMM(vector<unsigned char> &genome, int start, const int numSensors,
            const int numHidden, const int numMotors,
            const bool deterministic);
    ~HMM();

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;
    bool mDeterministic;

    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;
    vector<unsigned char> ins, outs;
    unsigned char numInputs, numOutputs;

    void update(unsigned char *currentStates, unsigned char *nextStates);
};

#endif  // SRC_HMM_H_
