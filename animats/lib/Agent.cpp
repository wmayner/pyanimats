// Agent.cpp

#include <vector>

#include "Agent.hpp"


Agent::Agent() {
    for (int i = 0; i < NUM_NODES; i++) {
        states[i] = 0;
        newStates[i] = 0;
    }
    hits = 0;
    hmmus.clear();
}

void Agent::setupEmptyAgent(int genomeSize) {
    genome.resize(genomeSize);
    for (int i = 0; i < genomeSize; i++) {
        genome[i] = 127;
    }
    setupPhenotype();
}

void Agent::setupPhenotype() {
    if (hmmus.size() != 0) {
        for (int i = 0; i < hmmus.size(); i++) {
            delete hmmus[i];
        }
    }
    hmmus.clear();
    HMM *hmmu;
    for (int i = 0; i < genome.size(); i++) {
        if ((genome[i] == 42) && (genome[(i + 1) %genome.size()] == 213)) {
            hmmu = new HMM(genome, i);
            hmmus.push_back(hmmu);
        }
    }
}

void Agent::resetBrain(void) {
    for (int i = 0; i < NUM_NODES; i++)
        states[i] = 0;
}

void Agent::updateStates(void) {
    for (int i = 0; i < hmmus.size(); i++) {
        hmmus[i]->update(&states[0], &newStates[0]);
    }
    for (int i = 0; i < NUM_NODES; i++) {
        states[i] = newStates[i];
        newStates[i] = 0;
    }
}

void Agent::injectStartCodons(int n) {
    for (int i = 0; i < genome.size(); i++)
        genome[i] = rand() & 255;
    for (int i = 0; i < n; i++) {
        int j = rand() % (genome.size() - 100);
        genome[j] = 42;
        genome[j + 1]= 213;
        for (int k = 2; k < 20; k++)
            genome[j + k] = rand() & 255;
    }
}
