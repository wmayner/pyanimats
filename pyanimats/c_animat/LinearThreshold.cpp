// LinearThreshold.cpp

#include "./LinearThreshold.hpp"


// Define start codon pair for this gate
// (Using 11, because 42 onward are used by Adami lab's MABE software)
unsigned char LinearThreshold::START_CODON_ONE = 11;
unsigned char LinearThreshold::START_CODON_TWO = 255 - START_CODON_ONE;


LinearThreshold::LinearThreshold(vector<unsigned char> &genome, int start,
        const int numSensors, const int numHidden, const int numMotors, const
        bool deterministic) {
    inputs.clear();
    outputs.clear();

    mNumSensors = numSensors;
    mNumHidden = numHidden;
    mNumMotors = numMotors;
    mNumNodes = numSensors + numHidden + numMotors;
    mDeterministic = deterministic;

    // This keeps track of where we are in the genome
    int scan = (start + 2) % (int)genome.size();

    // Maximum number of inputs is the number of non-motor nodes
    int maxInputs = mNumNodes - mNumMotors;
    // Number of possible outputs is the number of non-sensor nodes
    int maxOutputs = mNumNodes - mNumSensors;

    // At least one input is guaranteed
    numInputs = 1 + (genome[(scan++) % (int)genome.size()] % (maxInputs - 1));
    inputs.resize(numInputs);
    for (int i = 0; i < numInputs; i++)
        // Exclude motors from possible inputs
        inputs[i] = genome[(scan + i) % (int)genome.size()]
            % (mNumNodes - mNumMotors);
    // Move past the input codon
    scan += maxInputs;

    // There's always just 1 output
    outputs.resize(1);
    outputs[0] = mNumSensors + (genome[(scan + 1) % (int)genome.size()] % maxOutputs);
    // Move past the output codons
    scan += maxOutputs;

    // Get the threshold
    threshold = genome[(scan++) % (int)genome.size()] % numInputs;
}

void LinearThreshold::update(
        vector<unsigned char> &currentStates,
        vector<unsigned char> &nextStates) {
    // Count the number of inputs that are on
    int inputCount = 0;
    for (int i = 0; i < (int)inputs.size(); i++)
        inputCount += (currentStates[inputs[i]] & 1);
    // Activate output if count exceeds threshold
    if (inputCount > threshold) {
        // NOTE: Overwriting the output, rather than merging it with an OR,
        // ensures that each node effectively only recieves input from one
        // threshold gate (the last one in the genome that outputs to it)
        nextStates[outputs[0]] = 1;
    }
}

LinearThreshold::~LinearThreshold() {
    inputs.clear();
    outputs.clear();
}
