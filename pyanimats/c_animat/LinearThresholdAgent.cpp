// LinearThresholdAgent.cpp

#include "./LinearThresholdAgent.hpp"


void LinearThresholdAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    LinearThresholdGate *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == LinearThresholdGate::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 LinearThresholdGate::START_CODON_TWO)) {
            gate = new LinearThresholdGate(genome, i, mNumSensors, mNumHidden,
                    mNumMotors, mDeterministic);
            gates.push_back(gate);
        }
    }
}

void LinearThresholdAgent::injectStartCodons(int n) {
    injectStartCodons(n, LinearThresholdGate::START_CODON_ONE,
            LinearThresholdGate::START_CODON_TWO);
}

LinearThresholdAgent::~LinearThresholdAgent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

unsigned char LinearThresholdAgent::START_CODON_ONE = LinearThresholdGate::START_CODON_ONE;
unsigned char LinearThresholdAgent::START_CODON_TWO = LinearThresholdGate::START_CODON_TWO;
