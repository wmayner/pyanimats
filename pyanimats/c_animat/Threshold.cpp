// Threshold.cpp

#include "./Threshold.hpp"


Threshold::Threshold(vector<unsigned char> &genome, int start, const int
        numSensors, const int numHidden, const int numMotors) {
    inputs.clear();

    mNumSensors = numSensors;
    mNumHidden = numHidden;
    mNumMotors = numMotors;
    mNumNodes = numSensors + numHidden + numMotors;

    // This keeps track of where we are in the genome.
    int scan = (start + 2) % (int)genome.size();

    // Maximum number of inputs is the number of non-motor nodes.
    int maxInputs = mNumNodes - mNumMotors;
    // Number of possible outputs is the number of non-sensor nodes.
    int maxOutputs = mNumNodes - mNumSensors;

    // At least one input is guaranteed.
    numInputs = 1 + (genome[(scan++) % (int)genome.size()] % (maxInputs - 1));
    inputs.resize(numInputs);
    for (int i = 0; i < numInputs; i++)
        // Exclude motors from possible inputs.
        inputs[i] = genome[(scan + i) % (int)genome.size()]
            % (mNumNodes - mNumMotors);
    // Move past the input codon.
    scan += maxInputs;

    // Get the single output node.
    output = mNumSensors + (genome[(scan++) % (int)genome.size()] % maxOutputs);

    // Get the threshold.
    threshold = genome[(scan++) % (int)genome.size()] % numInputs;
}

void Threshold::update(unsigned char *currentStates, unsigned char *nextStates) {
    // Count the number of inputs that are on
    int inputCount = 0;
    for (int i = 0; i < (int)inputs.size(); i++)
        inputCount += (currentStates[inputs[i]] & 1);
    // Activate output if count exceeds threshold.
    if (inputCount > threshold) {
        nextStates[output] = 1;
    }
}

Threshold::~Threshold() {
    inputs.clear();
}
