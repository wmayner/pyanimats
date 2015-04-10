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
    vector<HMMU*> hmmus;
    vector<unsigned char> genome;
    Agent *ancestor;
    unsigned int nrPointingAtMe;
    unsigned char states[NUM_NODES], newStates[NUM_NODES];
    double fitness;
    vector<double> fitnesses;
    int ID;
    int born;
    int correct, incorrect;
    vector<int> numCorrectByPattern;

    Agent();
    ~Agent();
    void injectStartCodons();
    void setupEmptyAgent(int nucleotides);
    void inherit(Agent *parent, double mutationRate, int generation);
    void setupPhenotype();
    void resetBrain();
    void updateStates();
    void loadAgent(char* filename);
    void saveLogicTable(FILE *f);
    void saveLogicTableSingleAnimat(FILE *f);
    void saveGenome(FILE *f);
    void saveEdgeList(char *filename);
};

#endif  // SRC_AGENT_H_
