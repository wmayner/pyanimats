// AbstractGate.hpp

#ifndef _PyAnimats_AbstractGate_H
#define _PyAnimats_AbstractGate_H

#include <stdio.h>

#include <vector>

using std::vector;

// Abstract base class for different types of gates
class AbstractGate {
 public:
    virtual ~AbstractGate() = default;

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

#endif  // _PyAnimats_AbstractGate_H
