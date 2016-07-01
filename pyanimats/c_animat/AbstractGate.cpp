// AbstractGate.cpp

#include "./AbstractGate.hpp"


AbstractGate::AbstractGate(const int numSensors, const int numHidden,
        const int numMotors, const bool deterministic) {
    mNumSensors = numSensors;
    mNumHidden = numHidden;
    mNumMotors = numMotors;
    mNumNodes = numSensors + numHidden + numMotors;
    mDeterministic = deterministic;

    inputs.clear();
    outputs.clear();
}
