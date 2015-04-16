// Agent.h

#ifndef SRC_AGENT_H_
#define SRC_AGENT_H_

#include <vector>

#include "./constants.hpp"
#include "./HMM.hpp"

using std::vector;

class Agent {
 public:
    Agent(vector<unsigned char> genome);
    ~Agent();

    vector<HMM*> hmmus;
    vector<unsigned char> genome;
    int hits;
    unsigned char states[NUM_NODES], newStates[NUM_NODES];

    void injectStartCodons(int n);
    void resetState();
    void updateStates();
};

#endif  // SRC_AGENT_H_
