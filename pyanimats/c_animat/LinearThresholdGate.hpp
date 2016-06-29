// LinearThresholdGate.hpp

#ifndef ANIMAT_LINEARTHRESHOLDGATE_H_
#define ANIMAT_LINEARTHRESHOLDGATE_H_

#include <vector>

#include "./AbstractGate.hpp"

using std::vector;


class LinearThresholdGate: public AbstractGate {
 public:
    LinearThresholdGate(vector<unsigned char> &genome, int start,
            const int numSensors, const int numHidden, const int numMotors,
            const bool deterministic);
    ~LinearThresholdGate();

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE, START_CODON_TWO;

    int threshold;

    void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) override;
    void print() override;
};

#endif  // ANIMAT_LINEARTHRESHOLDGATE_H_
