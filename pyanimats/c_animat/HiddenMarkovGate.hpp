// HiddenMarkovGate.hpp

#ifndef _PyAnimats_HiddenMarkovGate_H
#define _PyAnimats_HiddenMarkovGate_H

#include <vector>

#include "./rng.hpp"
#include "./AbstractGate.hpp"

using std::vector;

class HiddenMarkovGate: public AbstractGate {
 public:
    HiddenMarkovGate(vector<unsigned char> &genome, int start,
            const int numSensors, const int numHidden, const int numMotors,
            const bool deterministic);
    ~HiddenMarkovGate();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE, START_CODON_TWO;

    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;

    void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) override;
    void print() override;
};

#endif  // _PyAnimats_HiddenMarkovGate_H
