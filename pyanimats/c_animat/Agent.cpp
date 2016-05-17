// Agent.cpp

#include "./Agent.hpp"

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
    hmms.clear();
}

Agent::~Agent() {
    for (int i = 0; i < (int)hmms.size(); i++) {
        delete hmms[i];
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
    for (int i = 0; i < (int)hmms.size(); i++) {
        hmms[i]->update(&states[0], &newStates[0]);
    }
    for (int i = 0; i < mNumNodes; i++) {
        states[i] = newStates[i];
        newStates[i] = 0;
    }
}

void Agent::generatePhenotype() {
    if (hmms.size() != 0) {
        for (int i = 0; i < (int)hmms.size(); i++) {
            delete hmms[i];
        }
    }
    hmms.clear();
    HMM *hmm;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == START_CODON_NUCLEOTIDE_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 START_CODON_NUCLEOTIDE_TWO)) {
            hmm = new HMM(genome, i, mNumSensors, mNumHidden, mNumMotors,
                    mDeterministic);
            hmms.push_back(hmm);
        }
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

void Agent::injectStartCodons(int n) {
    for (int i = 0; i < (int)genome.size(); i++)
        genome[i] = randCharInt();
    for (int i = 0; i < n; i++) {
        int j = randInt() % ((int)genome.size() - 100);

        // Start codon
        genome[j] = START_CODON_NUCLEOTIDE_ONE;
        genome[j + 1] = START_CODON_NUCLEOTIDE_TWO;

        for (int k = 2; k < 20; k++)
            genome[j + k] = randCharInt();
    }
}

vector< vector<int> > Agent::getEdges() {
    vector< vector<int> > edgeList;
    edgeList.clear();
    vector<int> edge;
    for (int i = 0; i < (int)hmms.size(); i++) {
        for (int j = 0; j < (int)hmms[i]->ins.size(); j++) {
            for (int k = 0; k < (int)hmms[i]->outs.size(); k++) {
                edge.clear();
                edge.resize(2);
                edge[0] = hmms[i]->ins[j];
                edge[1] = hmms[i]->outs[k];
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
