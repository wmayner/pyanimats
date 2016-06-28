// LinearThresholdAgent.hpp

#ifndef ANIMAT_LINEARTHRESHOLDAGENT_H_
#define ANIMAT_LINEARTHRESHOLDAGENT_H_

#include <vector>

#include "./AbstractAgent.hpp"
#include "./LinearThreshold.hpp"

using std::vector;

class LinearThresholdAgent: public AbstractAgent {
 public:
    LinearThresholdAgent(vector<unsigned char> genome, int numSensors, int
            numHidden, int numMotors, bool deterministic)
    : AbstractAgent(genome, numSensors, numHidden, numMotors, deterministic)
    {}
    ~LinearThresholdAgent();

    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    void generatePhenotype();

    using AbstractAgent::injectStartCodons;
    void injectStartCodons(int n);

};

#endif  // ANIMAT_LINEARTHRESHOLDAGENT_H_
