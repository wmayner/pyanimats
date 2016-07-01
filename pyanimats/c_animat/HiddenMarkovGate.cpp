// HiddenMarkovGate.cpp

#include "./HiddenMarkovGate.hpp"

// Define start codon pair for this gate
unsigned char HiddenMarkovGate::START_CODON_ONE = 42;
unsigned char HiddenMarkovGate::START_CODON_TWO = 255 - START_CODON_ONE;


HiddenMarkovGate::HiddenMarkovGate(vector<unsigned char> &genome, int start,
        const int numSensors, const int numHidden, const int numMotors,
        const bool deterministic)
    : AbstractGate(numSensors, numHidden, numMotors, deterministic) {

    // This keeps track of where we are in the genome.
    int scan = (start + 2) % (int)genome.size();

    numInputs = 1 + (genome[(scan++) % (int)genome.size()] & 3);
    numOutputs = 1 + (genome[(scan++) % (int)genome.size()] & 3);
    inputs.resize(numInputs);
    outputs.resize(numOutputs);

    for (int i = 0; i < numInputs; i++)
        // Exclude motors from possible inputs.
        inputs[i] = genome[(scan + i) % (int)genome.size()]
                % (mNumNodes - mNumMotors);
    for (int i = 0; i < numOutputs; i++)
        // Exclude sensors from possible outputs.
        outputs[i] = (genome[(scan + 4 + i) % (int)genome.size()]
                % (mNumNodes - mNumSensors)) + mNumSensors;

    // Probabilities begin after the input and output codons, which are maximum
    // 4 nucleotides long each, and an "intron" region of length 8 (for
    // consistency with previous versions.)
    scan += 16;

    // Number of rows
    int M = 1 << numInputs;
    // Number of columns
    int N = 1 << numOutputs;

    hmm.resize(M);
    sums.resize(M);

    if (mDeterministic) {
        for (int i = 0; i < M; i++) {
            hmm[i].resize(N);
            int largestValueInRow = 0;
            int largestValueInRowIndex = 0;
            for (int j = 0; j < (N); j++) {
                hmm[i][j] = 0;
                int currentValue = genome[(scan + j + (N * i)) % (int)genome.size()];
                if (currentValue > largestValueInRow) {
                    largestValueInRow = currentValue;
                    largestValueInRowIndex = j;
                }
            }
            hmm[i][largestValueInRowIndex] = 255;
            sums[i] = 255;
        }
    } else {
        for (int i = 0; i < M; i++) {
            hmm[i].resize(N);
            for (int j = 0; j < N; j++) {
                hmm[i][j] = genome[(scan + j + (N * i)) % (int)genome.size()];
                // Don't allow zero-entries
                // TODO(wmayner) why?
                if (hmm[i][j] == 0) hmm[i][j] = 1;
                sums[i] += hmm[i][j];
            }
        }
    }
}

void HiddenMarkovGate::update(vector<unsigned char> &currentStates,
        vector<unsigned char> &nextStates) {
    // Encode the given states as an integer to index into the TPM
    int pastStateIndex = 0;
    for (int i = 0; i < (int)inputs.size(); i++)
        pastStateIndex = (pastStateIndex << 1) + ((currentStates[inputs[i]]) & 1);
    // Get the next state
    int nextStateIndex = 0;
    if (mDeterministic) {
        // Find the index of the 1 in this row
        while (1 > hmm[pastStateIndex][nextStateIndex]) {
            nextStateIndex++;
        }
    } else {
        // Randomly pick a column index with probabilities weighted by the entries
        // in the column.
        // TODO this is [1, sums[pastStateIndex] - 1]; is that what we want?
        int r = 1 + (randInt() % (sums[pastStateIndex] - 1));
        while (r > hmm[pastStateIndex][nextStateIndex]) {
            // Decrease the random threshold because it's given that we didn't
            // pick column nextStateIndex, which we would have with probability
            // hmm[pastStateIndex][nextStateIndex].
            r -= hmm[pastStateIndex][nextStateIndex];
            nextStateIndex++;
        }
    }
    // The index of the column we chose is the next state (we take the its bits
    // as the next states of individual nodes)
    for (int i = 0; i < (int)outputs.size(); i++) {
        nextStates[outputs[i]] |= (nextStateIndex >> i) & 1;
    }
}

HiddenMarkovGate::~HiddenMarkovGate() {
    hmm.clear();
    sums.clear();
    inputs.clear();
    outputs.clear();
}

void HiddenMarkovGate::print() {
    printf("\n------------------");
    printf("\nHidden Markov Gate");
    printf("\n------------------");
    printf("\n   inputs: %i:\t[", numInputs);
    for (int i = 0; i < ((int)inputs.size() - 1); i++) {
        printf("%i, ", inputs[i]);
    }
    printf("%i]", inputs[(int)inputs.size() - 1]);
    printf("\n  outputs: %i:\t[", numOutputs);
    for (int i = 0; i < ((int)outputs.size() - 1); i++) {
        printf("%i, ", outputs[i]);
    }
    printf("%i]", outputs[(int)outputs.size() - 1]);
}
