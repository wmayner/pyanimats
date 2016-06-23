// HMM.h

#ifndef ANIMAT_HMM_H_
#define ANIMAT_HMM_H_

#include <vector>

#include "./rng.hpp"
#include "./Gate.hpp"

using std::vector;

class HMM: public Gate {
 public:
    HMM(vector<unsigned char> &genome, int start, const int numSensors,
            const int numHidden, const int numMotors,
            const bool deterministic);
    ~HMM();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;
    bool mDeterministic;

    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;
    vector<unsigned char> inputs, outputs;
    unsigned char numInputs, numOutputs;

    void update(vector<unsigned char> &currentStates, vector<unsigned char>
            &nextStates);
};

#endif  // ANIMAT_HMM_H_
