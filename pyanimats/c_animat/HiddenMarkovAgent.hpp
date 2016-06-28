// HiddenMarkovAgent.hpp

#ifndef ANIMAT_HIDDENMARKOVAGENT_H_
#define ANIMAT_HIDDENMARKOVAGENT_H_

#include <vector>

#include "./AbstractAgent.hpp"
#include "./HMM.hpp"

using std::vector;

class HMMAgent: public AbstractAgent {
 public:
    HMMAgent(vector<unsigned char> genome, int numSensors, int numHidden, int
            numMotors, bool deterministic)
    : AbstractAgent(genome, numSensors, numHidden, numMotors, deterministic)
    {}
    ~HMMAgent();

    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    void generatePhenotype();

    using AbstractAgent::injectStartCodons;
    void injectStartCodons(int n);
};

#endif  // ANIMAT_HIDDENMARKOVAGENT_H_
