// Threshold.hpp

#ifndef ANIMAT_LINEAR_THRESHOLD_H_
#define ANIMAT_LINEAR_THRESHOLD_H_

#include <vector>

#include "./Gate.hpp"

using std::vector;

class LinearThreshold: public Gate {
 public:
    LinearThreshold(vector<unsigned char> &genome, int start, const int
            numSensors, const int numHidden, const int numMotors,
            const bool deterministic);
    ~LinearThreshold();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;
    bool mDeterministic;

    unsigned char numInputs;
    vector<unsigned char> inputs;

    vector<unsigned char> outputs;

    int threshold;

    void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates);
};

#endif  // ANIMAT_LINEAR_THRESHOLD_H_
