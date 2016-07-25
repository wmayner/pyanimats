// LinearThresholdGate.cpp

#include "./LinearThresholdGate.hpp"


// Define start codon pair for this gate
// (Using 11, because 42 onward are used by Adami lab's MABE software)
unsigned char LinearThresholdGate::START_CODON_ONE = 11;
unsigned char LinearThresholdGate::START_CODON_TWO = 255 - START_CODON_ONE;


LinearThresholdGate::LinearThresholdGate(vector<unsigned char> &genome,
        int start, const int numSensors, const int numHidden,
        const int numMotors, const bool deterministic)
    : AbstractGate(numSensors, numHidden, numMotors, deterministic) {

    // This keeps track of where we are in the genome
    int scan = (start + 2) % (int)genome.size();

    // Maximum number of inputs is the number of non-motor nodes
    int maxInputs = mNumNodes - mNumMotors;
    // Number of possible outputs is the number of non-sensor nodes
    int maxOutputs = mNumNodes - mNumSensors;

    // Get the threshold
    threshold = genome[(scan++) % (int)genome.size()] % maxInputs;

    // At least one input is guaranteed
    numInputs = 1 + (genome[(scan++) % (int)genome.size()] % maxInputs);
    inputs.resize(numInputs);
    // At least one output is guaranteed
    numOutputs = 1 + (genome[(scan++) % (int)genome.size()] % maxOutputs);
    outputs.resize(numOutputs);

    // Get inputs
    //
    // This vector keeps track of which inputs we have not already chosen, so
    // we don't get duplicated inputs
    // NOTE: This excludes motors from possible inputs by considering only
    // values between 0 and `maxInputs`
    vector<int> available;
    available.clear();
    available.resize(maxInputs);
    for (int i = 0; i < maxInputs; i++)
        available[i] = i;
    int input;
    for (int i = 0; i < numInputs; i++) {
        input = genome[(scan + i) % (int)genome.size()] % (int)available.size();
        inputs[i] = available[input];
        available.erase(available.begin() + input);
    }
    // Move past the input codon
    scan += maxInputs;

    // Get outputs
    for (int i = 0; i < numOutputs; i++)
        // Exclude sensors from possible outputs.
        outputs[i] = (genome[(scan + i) % (int)genome.size()] % maxOutputs)
            + mNumSensors;
}

void LinearThresholdGate::update(
        vector<unsigned char> &currentStates,
        vector<unsigned char> &nextStates) {
    // Count the number of inputs that are on
    int inputCount = 0;
    for (int i = 0; i < (int)inputs.size(); i++)
        inputCount += (currentStates[inputs[i]] & 1);
    // Activate outputs if count exceeds threshold
    // NOTE: Overwriting the output, rather than merging it with an OR,
    // ensures that each node effectively only recieves input from one
    // threshold gate (the last one in the genome that outputs to it)
    if (inputCount > threshold) {
        for (int i = 0; i < (int)outputs.size(); i++)
            nextStates[outputs[i]] = 1;
    }
    else {
        for (int i = 0; i < (int)outputs.size(); i++)
            nextStates[outputs[i]] = 0;
    }
}

LinearThresholdGate::~LinearThresholdGate() {
    inputs.clear();
    outputs.clear();
}

void LinearThresholdGate::print() {
    printf("\n---------------------");
    printf("\nLinear Threshold Gate");
    printf("\n---------------------");
    printf("\n  threshold: %i", threshold);
    printf("\n     inputs: %i:\t[", numInputs);
    for (int i = 0; i < ((int)inputs.size() - 1); i++) {
        printf("%i, ", inputs[i]);
    }
    printf("%i]", inputs[(int)inputs.size() - 1]);
    printf("\n    outputs: %i:\t[", numOutputs);
    for (int i = 0; i < ((int)outputs.size() - 1); i++) {
        printf("%i, ", outputs[i]);
    }
    printf("%i]", outputs[(int)outputs.size() - 1]);
}
