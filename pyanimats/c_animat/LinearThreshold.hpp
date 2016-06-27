// LinearThreshold.hpp

#ifndef ANIMAT_LINEAR_THRESHOLD_H_
#define ANIMAT_LINEAR_THRESHOLD_H_

#include <vector>

#include "./Gate.hpp"

using std::vector;


class LinearThreshold: public Gate {
 public:
    LinearThreshold(vector<unsigned char> &genome, int start,
            const int numSensors, const int numHidden, const int numMotors,
            const bool deterministic);
    ~LinearThreshold();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE, START_CODON_TWO;

    int threshold;

    void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) override;
    void print() override;
};

#endif  // ANIMAT_LINEAR_THRESHOLD_H_
