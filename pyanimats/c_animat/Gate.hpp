// Gate.hpp

#ifndef ANIMAT_GATE_H_
#define ANIMAT_GATE_H_

#include <vector>

using std::vector;

// Abstract base class for different types of gates
class Gate {
 public:
    virtual ~Gate() = default;

    int mNumHidden;
    int mNumMotors;
    int mNumSensors;
    int mNumNodes;
    bool mDeterministic;

    unsigned char numInputs, numOutputs;
    vector<unsigned char> inputs, outputs;

    virtual void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) = 0;
    virtual void print() = 0;
};

#endif  // ANIMAT_GATE_H_
