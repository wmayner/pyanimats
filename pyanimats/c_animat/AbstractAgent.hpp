// AbstractAgent.hpp

#pragma once

#include <vector>

#include "./constants.hpp"
#include "./rng.hpp"
#include "./AbstractGate.hpp"

using std::vector;

// Abstract base class
class AbstractAgent {
 public:
    AbstractAgent(vector<unsigned char> genome, int numSensors, int numHidden, int
            numMotors, bool deterministic);
    virtual ~AbstractAgent() = default;

    // Note on naming: the `m` prefix indicates a member variable
    int mNumSensors;
    int mNumHidden;
    int mNumMotors;
    int mNumNodes;
    int mNumStates;
    int mBodyLength;
    bool mDeterministic;

    vector<AbstractGate*> gates;

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
    void printGates();

    virtual void generatePhenotype() = 0;
};
