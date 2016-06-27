// HMM.h

#ifndef ANIMAT_HMM_H_
#define ANIMAT_HMM_H_

#include <vector>

#include "./rng.hpp"
#include "./Gate.hpp"

using std::vector;

class HMM: public Gate {
 public:
    HMM(vector<unsigned char> &genome, int start, const int numSensors,
            const int numHidden, const int numMotors,
            const bool deterministic);
    ~HMM();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE, START_CODON_TWO;

    vector< vector<unsigned char> > hmm;
    vector<unsigned int> sums;

    void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) override;
    void print() override;
};

#endif  // ANIMAT_HMM_H_
