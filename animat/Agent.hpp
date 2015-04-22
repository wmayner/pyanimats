// Agent.h

#ifndef SRC_AGENT_H_
#define SRC_AGENT_H_

#include <stdlib.h>  // srand, rand
#include <vector>

#include "./constants.hpp"
#include "./HMM.hpp"

using std::vector;

class Agent {
 public:
    Agent(vector<unsigned char> genome);
    ~Agent();

    vector<HMM*> hmms;
    vector<unsigned char> genome;
    int hits;
    unsigned char states[NUM_NODES], newStates[NUM_NODES];

    void injectStartCodons(int n);
    void resetState();
    void updateStates();
    void generatePhenotype();
    void mutateGenome(double mutProb, double dupProb, double delProb, int
            minGenomeLength, int maxGenomeLength);
};

#endif  // SRC_AGENT_H_
