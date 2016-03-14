// Agent.h

#ifndef ANIMAT_AGENT_H_
#define ANIMAT_AGENT_H_

#include <stdlib.h>  // srand, rand
#include <vector>

#include "./constants.hpp"
#include "./HMM.hpp"
#include "./rng.hpp"

using std::vector;

class Agent {
 public:
    Agent(vector<unsigned char> genome, int numSensors, int numHidden,
        int numMotors, bool deterministic);
    ~Agent();

    // Note on naming: the `m` prefix indicates a member variable
    int mNumSensors;
    int mNumHidden;
    int mNumMotors;
    int mNumNodes;
    int mNumStates;
    int mBodyLength;
    bool mDeterministic;

    vector<HMM*> hmms;
    vector<unsigned char> genome;
    // TODO(wmayner) change these to bool?
    vector<unsigned char> states;
    vector<unsigned char> newStates;

    void injectStartCodons(int n);
    void resetState();
    void updateStates();
    void generatePhenotype();
    void mutateGenome(double mutProb, double dupProb, double delProb, int
        minGenomeLength, int maxGenomeLength, int minDupDelLength,
        int maxDupDelLength);
    vector< vector<int> > getEdges();
    vector< vector<bool> > getTransitions();
};

#endif  // ANIMAT_AGENT_H_
