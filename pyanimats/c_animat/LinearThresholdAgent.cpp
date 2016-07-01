// LinearThresholdAgent.cpp

#include "./LinearThresholdAgent.hpp"


void LinearThresholdAgent::generatePhenotype() {
    if (gates.size() != 0) {
        for (int i = 0; i < (int)gates.size(); i++) {
            delete gates[i];
        }
    }
    gates.clear();
    LinearThresholdGate *gate;
    for (int i = 0; i < (int)genome.size(); i++) {
        if ((genome[i] == LinearThresholdGate::START_CODON_ONE) &&
                (genome[(i + 1) % (int)genome.size()] ==
                 LinearThresholdGate::START_CODON_TWO)) {
            gate = new LinearThresholdGate(genome, i, mNumSensors, mNumHidden,
                    mNumMotors, mDeterministic);
            gates.push_back(gate);
        }
    }
}

void LinearThresholdAgent::injectStartCodons(int n) {
    injectStartCodons(n, LinearThresholdGate::START_CODON_ONE,
            LinearThresholdGate::START_CODON_TWO);
}

vector< vector<int> > LinearThresholdAgent::getEdges() {
    vector< vector<int> > edgeList;
    edgeList.clear();
    vector<int> edge;
    vector<unsigned char> inputs;
    bool isInput;
    for (int node = 0; node < mNumNodes; node++) {
        // A node's inputs are the inputs of the last gate that wrote to it
        // (the last one in the genome that outputs to it)
        for (int i = 0; i < (int)gates.size(); i++) {
            // Does the gate output to this node?
            isInput = false;
            for (int j = 0; j < (int)gates[i]->outputs.size(); j++)
                if (gates[i]->outputs[j] == node)
                    isInput = true;
            // If so, overwrite inputs
            if (isInput) {
                inputs = gates[i]->inputs;
            }
        }
        // Add an edge from each input node to this node
        for (unsigned long i = 0; i < inputs.size(); i++) {
            edge.clear();
            edge.resize(2);
            edge[0] = inputs[i];
            edge[1] = node;
            edgeList.push_back(edge);
        }
    }
    return edgeList;
}

LinearThresholdAgent::~LinearThresholdAgent() {
    for (int i = 0; i < (int)gates.size(); i++) {
        delete gates[i];
    }
}

unsigned char LinearThresholdAgent::START_CODON_ONE = LinearThresholdGate::START_CODON_ONE;
unsigned char LinearThresholdAgent::START_CODON_TWO = LinearThresholdGate::START_CODON_TWO;
