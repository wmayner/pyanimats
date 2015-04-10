// Agent.cpp

#include <vector>

#include "Agent.hpp"


Agent::Agent() {
    nrPointingAtMe = 1;
    ancestor = NULL;
    for (int i = 0; i < NUM_NODES; i++) {
        states[i] = 0;
        newStates[i] = 0;
    }
    ID = masterID++;
    hmmus.clear();
}

void Agent::setupEmptyAgent(int nucleotides) {
    genome.resize(nucleotides);
    for (int i = 0; i < nucleotides; i++) {
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
    HMMU *hmmu;
    for (int i = 0; i < genome.size(); i++) {
        if ((genome[i] == 42) && (genome[(i + 1) %genome.size()] == 213)) {
            hmmu = new HMMU(genome, i);
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

void Agent::injectStartCodons() {
    for (int i = 0; i < genome.size(); i++)
        genome[i] = rand() & 255;
    for (int i = 0; i < 4; i++) {
        int j = rand() % (genome.size() - 100);
        genome[j] = 42;
        genome[j + 1]= 213;
        for (int k = 2; k < 20; k++)
            genome[j + k] = rand() & 255;
    }
}

void Agent::inherit(Agent *parent, double mutationRate, int generation) {
    int nucleotides = parent->genome.size();
    vector<unsigned char> buffer;
    born = generation;
    ancestor = parent;
    parent->nrPointingAtMe++;
    genome.clear();
    genome.resize(parent->genome.size());
    // Mutation
    for (int i = 0; i < nucleotides; i++) {
        if (randDouble < mutationRate) {
            genome[i] = rand() & 255;
        } else {
            genome[i] = parent->genome[i];
        }
    }
    if (mutationRate != 0.0) {
        if ((randDouble < 0.05) && (genome.size() < 10000)) {
            int w = 15 + rand() & 511;
            // Duplication
            int s = rand() % ((int)genome.size() - w);
            int o = rand() % (int)genome.size();
            buffer.clear();
            buffer.insert(buffer.begin(), genome.begin() + s,
                    genome.begin() + s + w);
            genome.insert(genome.begin() + o, buffer.begin(), buffer.end());
        }
        if ((randDouble < 0.02) && (genome.size() > 1000)) {
            // Deletion
            int w = 15 + rand() & 511;
            int s = rand() % ((int)genome.size() - w);
            genome.erase(genome.begin() + s, genome.begin() + s + w);
        }
    }
    setupPhenotype();
    fitness = 0.0;
}

Agent::~Agent() {
    for (int i = 0; i < hmmus.size(); i++) {
        delete hmmus[i];
    }
    if (ancestor != NULL) {
        ancestor->nrPointingAtMe--;
        if (ancestor->nrPointingAtMe == 0)
            delete ancestor;
    }
}

