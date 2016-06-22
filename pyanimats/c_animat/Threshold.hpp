// Threshold.hpp

#ifndef ANIMAT_THRESHOLD_H_
#define ANIMAT_THRESHOLD_H_

#include <vector>

using std::vector;

class Threshold {
 public:
    Threshold(vector<unsigned char> &genome, int start, const int numSensors,
            const int numHidden, const int numMotors);
    ~Threshold();

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;

    unsigned char numInputs;
    vector<unsigned char> inputs;

    unsigned char output;

    int threshold;

    void update(unsigned char *currentStates, unsigned char *nextStates);
};

#endif  // THRESHOLD_HMM_H_
