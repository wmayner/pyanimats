// Agent.h

#ifndef SRC_AGENT_H_
#define SRC_AGENT_H_

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "./constants.hpp"
#include "./HMM.hpp"

using std::vector;

static int masterID = 0;

class Agent {
 public:
    Agent();
    ~Agent();

    vector<HMM*> hmmus;
    vector<unsigned char> genome;
    Agent *ancestor;
    unsigned int nrPointingAtMe;
    unsigned char states[NUM_NODES], newStates[NUM_NODES];
    int ID;
    int hits;

    void setupEmptyAgent(int nucleotides);
    void setupPhenotype();
    void injectStartCodons(int n);
    void resetBrain();
    void updateStates();
};

#endif  // SRC_AGENT_H_
