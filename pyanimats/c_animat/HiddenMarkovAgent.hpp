// HiddenMarkovAgent.hpp

#ifndef ANIMAT_HIDDENMARKOVAGENT_H_
#define ANIMAT_HIDDENMARKOVAGENT_H_

#include <vector>

#include "./AbstractAgent.hpp"
#include "./HiddenMarkovGate.hpp"

using std::vector;

class HiddenMarkovAgent: public AbstractAgent {
 public:
    HiddenMarkovAgent(vector<unsigned char> genome, int numSensors,
            int numHidden, int numMotors, bool deterministic)
    : AbstractAgent(genome, numSensors, numHidden, numMotors, deterministic)
    {}
    ~HiddenMarkovAgent();

    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    void generatePhenotype();

    using AbstractAgent::injectStartCodons;
    void injectStartCodons(int n);
};

#endif  // ANIMAT_HIDDENMARKOVAGENT_H_
