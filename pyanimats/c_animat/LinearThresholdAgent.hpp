// LinearThresholdAgent.hpp

#pragma once

#include <vector>

#include "./AbstractAgent.hpp"
#include "./LinearThresholdGate.hpp"

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

    vector< vector<int> > getEdges();
};
