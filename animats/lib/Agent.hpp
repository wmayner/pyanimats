// Agent.h

#ifndef SRC_AGENT_H_
#define SRC_AGENT_H_

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "./constants.hpp"
#include "./HMM.hpp"

using std::vector;

class Agent {
 public:
    Agent();
    ~Agent();

    vector<HMM*> hmmus;
    vector<unsigned char> genome;
    int hits;
    unsigned char states[NUM_NODES], newStates[NUM_NODES];

    void setupEmptyAgent(int nucleotides);
    void setupPhenotype();
    void injectStartCodons(int n);
    void resetState();
    void updateStates();
};

#endif  // SRC_AGENT_H_
