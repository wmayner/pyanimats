// HiddenMarkovAgent.cpp

#include "./HiddenMarkovAgent.hpp"


void HMMAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    HMM *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == HMM::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 HMM::START_CODON_TWO)) {
            gate = new HMM(genome, i, mNumSensors, mNumHidden, mNumMotors,
                    mDeterministic);
            gates.push_back(gate);
        }
    }
}

void HMMAgent::injectStartCodons(int n) {
    injectStartCodons(n, HMM::START_CODON_ONE, HMM::START_CODON_TWO);
}

HMMAgent::~HMMAgent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

unsigned char HMMAgent::START_CODON_ONE = HMMAgent::START_CODON_ONE;
unsigned char HMMAgent::START_CODON_TWO = HMMAgent::START_CODON_TWO;
