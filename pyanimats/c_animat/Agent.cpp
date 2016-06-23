// Agent.cpp

#include "./Agent.hpp"


// Agent base class
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent::Agent(vector<unsigned char> genome, int numSensors, int numHidden,
        int numMotors, bool deterministic) : genome(genome) {
    mNumSensors = numSensors;
    mNumHidden = numHidden;
    // Note: only the last 2 motors have an effect. There must be > 2 motors or
    // none.
    mNumMotors = numMotors;
    mNumNodes = mNumSensors + mNumHidden + mNumMotors;
    mNumStates = 1 << mNumNodes;
    mBodyLength = std::max(MIN_BODY_LENGTH, mNumSensors);
    mDeterministic = deterministic;

    states.resize(mNumNodes);
    newStates.resize(mNumNodes);
    for (int i = 0; i < mNumNodes; i++) {
        states[i] = 0;
        newStates[i] = 0;
    }
    gates.clear();
}

Agent::~Agent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

int Agent::getAction() {
    if (mNumMotors > 0) {
        return (states[mNumNodes - 2] << 1) + states[mNumNodes - 1];
    }
    else return 0;
}

void Agent::resetState() {
    for (int i = 0; i < mNumNodes; i++)
        states[i] = 0;
}

void Agent::updateStates() {
    for (int i = 0; i < (int)gates.size(); i++) {
        gates[i]->update(states, newStates);
    }
    for (int i = 0; i < mNumNodes; i++) {
        states[i] = newStates[i];
        newStates[i] = 0;
    }
}


void Agent::mutateGenome(double mutProb, double dupProb, double delProb,
        int minGenomeLength, int maxGenomeLength, int minDupDelLength,
        int maxDupDelLength) {
    // Mutation
    for (int i = 0; i < (int)genome.size(); i++) {
        if (randDouble() < mutProb) {
            genome[i] = randCharInt();
        }
    }
    // Duplication
    if ((randDouble() < dupProb) && ((int)genome.size() < maxGenomeLength)) {
        int width = (minDupDelLength + randInt()) & maxDupDelLength;
        int start = randInt() % ((int)genome.size() - width);
        int insert = randInt() % (int)genome.size();
        vector<unsigned char> buffer;
        buffer.clear();
        buffer.insert(buffer.begin(), genome.begin() + start, genome.begin() +
                start + width);
        genome.insert(genome.begin() + insert, buffer.begin(), buffer.end());
    }
    // Deletion
    if ((randDouble() < delProb) && ((int)genome.size() > minGenomeLength)) {
        int width = (minDupDelLength + randInt()) & maxDupDelLength;
        int start = randInt() % ((int)genome.size() - width);
        genome.erase(genome.begin() + start, genome.begin() + start + width);
    }
}

void Agent::injectStartCodons(int n, unsigned char codon_one,
        unsigned char codon_two) {
    for (int i = 0; i < (int)genome.size(); i++)
        genome[i] = randCharInt();
    for (int i = 0; i < n; i++) {
        int j = randInt() % ((int)genome.size() - 100);

        // Start codon
        genome[j] = codon_one;
        genome[j + 1] = codon_two;

        for (int k = 2; k < 20; k++)
            genome[j + k] = randCharInt();
    }
}

vector< vector<int> > Agent::getEdges() {
    vector< vector<int> > edgeList;
    edgeList.clear();
    vector<int> edge;
    for (int i = 0; i < (int)gates.size(); i++) {
        for (int j = 0; j < (int)gates[i]->inputs.size(); j++) {
            for (int k = 0; k < (int)gates[i]->outputs.size(); k++) {
                edge.clear();
                edge.resize(2);
                edge[0] = gates[i]->inputs[j];
                edge[1] = gates[i]->outputs[k];
                edgeList.push_back(edge);
            }
        }
    }
    return edgeList;
}


vector< vector<bool> > Agent::getTransitions() {
    // Save animat's original state.
    unsigned char initial_states[mNumNodes];
    for (int i = 0; i < mNumNodes; i++) {
        initial_states[i] = states[i];
    }
    vector< vector<bool> > tpm;
    tpm.clear();
    tpm.resize(mNumStates);
    for (int i = 0; i < mNumStates; i++) {
        // Set animat to the ith state (using LOLI mapping from states to
        // integers).
        for (int j = 0; j < mNumNodes; j++) {
            states[j] = (i >> j) & 1;
        }
        // Update the state to get the transition and record it.
        updateStates();
        tpm[i].resize(mNumNodes);
        for (int j = 0; j < mNumNodes; j++) {
            tpm[i][j] = states[j];
        }
    }
    // Return animat to its original state.
    for (int i = 0; i < mNumNodes; i++) {
        states[i] = initial_states[i];
    }
    return tpm;
}


// HMMAgent
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void HMMAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    HMM *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == HMM::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 HMM::START_CODON_TWO)) {
            gate = new HMM(genome, i, mNumSensors, mNumHidden, mNumMotors,
                    mDeterministic);
            gates.push_back(gate);
        }
    }
}

void HMMAgent::injectStartCodons(int n) {
    this->injectStartCodons(n, HMM::START_CODON_ONE, HMM::START_CODON_TWO);
}


// LinearThresholdAgent
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void LinearThresholdAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    LinearThreshold *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == LinearThreshold::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 LinearThreshold::START_CODON_TWO)) {
            gate = new LinearThreshold(genome, i, mNumSensors, mNumHidden,
                    mNumMotors, mDeterministic);
            gates.push_back(gate);
        }
    }
}

void LinearThresholdAgent::injectStartCodons(int n) {
    this->injectStartCodons(n, LinearThreshold::START_CODON_ONE,
            LinearThreshold::START_CODON_TWO);
}
