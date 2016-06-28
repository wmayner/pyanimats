// LinearThresholdAgent.cpp

#include "./LinearThresholdAgent.hpp"


void LinearThresholdAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    LinearThreshold *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == LinearThreshold::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 LinearThreshold::START_CODON_TWO)) {
            gate = new LinearThreshold(genome, i, mNumSensors, mNumHidden,
                    mNumMotors, mDeterministic);
            gates.push_back(gate);
        }
    }
}

void LinearThresholdAgent::injectStartCodons(int n) {
    injectStartCodons(n, LinearThreshold::START_CODON_ONE,
            LinearThreshold::START_CODON_TWO);
}

LinearThresholdAgent::~LinearThresholdAgent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

unsigned char LinearThresholdAgent::START_CODON_ONE = LinearThreshold::START_CODON_ONE;
unsigned char LinearThresholdAgent::START_CODON_TWO = LinearThreshold::START_CODON_TWO;
