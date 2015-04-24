// mutate.h

#ifndef SRC_MUTATE_H_
#define SRC_MUTATE_H_

#include <stdlib.h>  // srand, rand
#include <vector>

using std::vector;

/* void mutateGenome(vector<unsigned char> genome, double mutProb, double dupProb, */
/*         double delProb, int minGenomeLength, int maxGenomeLength); */
vector<unsigned char> mutateGenome(vector<unsigned char> genome, double
        mutProb, double dupProb, double delProb, int minGenomeLength, int
        maxGenomeLength);

#endif  // SRC_MUTATE_H_
