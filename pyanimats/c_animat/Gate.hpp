// Gate.hpp

#ifndef ANIMAT_GATE_H_
#define ANIMAT_GATE_H_

#include <vector>

using std::vector;

// Abstract base class for different types of gates
class Gate {
 public:
    virtual ~Gate() = 0;

    vector<unsigned char> inputs, outputs;

    // Start codon pair for this gate
    static unsigned char START_CODON_ONE;
    static unsigned char START_CODON_TWO;

    virtual void update(vector<unsigned char> &currentStates,
            vector<unsigned char> &nextStates) = 0;
};

inline Gate::~Gate() {}

#endif  // ANIMAT_GATE_H_
