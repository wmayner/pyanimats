// Agent.h

#ifndef ANIMAT_AGENT_H_
#define ANIMAT_AGENT_H_

#include <stdlib.h>  // srand, rand
#include <vector>

#include "./constants.hpp"
#include "./rng.hpp"
#include "./Gate.hpp"
#include "./HMM.hpp"
#include "./LinearThreshold.hpp"

using std::vector;

// Base class
class Agent {
 public:
    Agent(vector<unsigned char> genome, int numSensors, int numHidden, int
            numMotors, bool deterministic);
    virtual ~Agent() = default;

    // Note on naming: the `m` prefix indicates a member variable
    int mNumSensors;
    int mNumHidden;
    int mNumMotors;
    int mNumNodes;
    int mNumStates;
    int mBodyLength;
    bool mDeterministic;

    vector<Gate*> gates;

    vector<unsigned char> genome;
    // TODO(wmayner) change these to bool?
    vector<unsigned char> states;
    vector<unsigned char> newStates;

    int getAction();
    void resetState();
    void updateStates();
    void injectStartCodons(int n, unsigned char codon_one,
            unsigned char codon_two);
    void mutateGenome(double mutProb, double dupProb, double delProb, int
        minGenomeLength, int maxGenomeLength, int minDupDelLength,
        int maxDupDelLength);
    vector< vector<int> > getEdges();
    vector< vector<bool> > getTransitions();

    virtual void generatePhenotype() = 0;
};


class HMMAgent: public Agent {
 public:
    HMMAgent(vector<unsigned char> genome, int numSensors, int numHidden, int
            numMotors, bool deterministic)
    : Agent(genome, numSensors, numHidden, numMotors, deterministic)
    {}
    ~HMMAgent();

    using Agent::injectStartCodons;
    void injectStartCodons(int n);

    void generatePhenotype();
};


class LinearThresholdAgent: public Agent {
 public:
    LinearThresholdAgent(vector<unsigned char> genome, int numSensors, int
            numHidden, int numMotors, bool deterministic)
    : Agent(genome, numSensors, numHidden, numMotors, deterministic)
    {}
    ~LinearThresholdAgent();

    using Agent::injectStartCodons;
    void injectStartCodons(int n);

    void generatePhenotype();
};

#endif  // ANIMAT_AGENT_H_
