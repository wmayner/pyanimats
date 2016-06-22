// Threshold.hpp

#ifndef ANIMAT_LINEAR_THRESHOLD_H_
#define ANIMAT_LINEAR_THRESHOLD_H_

#include <vector>

using std::vector;

class LinearThreshold {
 public:
    LinearThreshold(vector<unsigned char> &genome, int start, const int
            numSensors, const int numHidden, const int numMotors);
    ~LinearThreshold();

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;

    unsigned char numInputs;
    vector<unsigned char> inputs;

    vector<unsigned char> outputs;

    int threshold;

    void update(unsigned char *currentStates, unsigned char *nextStates);
};

#endif  // ANIMAT_LINEAR_THRESHOLD_H_
