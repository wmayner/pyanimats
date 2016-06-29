// HiddenMarkovAgent.cpp

#include "./HiddenMarkovAgent.hpp"


void HiddenMarkovAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    HiddenMarkovGate *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == HiddenMarkovGate::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 HiddenMarkovGate::START_CODON_TWO)) {
            gate = new HiddenMarkovGate(genome, i, mNumSensors, mNumHidden,
                    mNumMotors, mDeterministic);
            gates.push_back(gate);
        }
    }
}

void HiddenMarkovAgent::injectStartCodons(int n) {
    injectStartCodons(n, HiddenMarkovGate::START_CODON_ONE,
            HiddenMarkovGate::START_CODON_TWO);
}

HiddenMarkovAgent::~HiddenMarkovAgent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

unsigned char HiddenMarkovAgent::START_CODON_ONE = HiddenMarkovAgent::START_CODON_ONE;
unsigned char HiddenMarkovAgent::START_CODON_TWO = HiddenMarkovAgent::START_CODON_TWO;
