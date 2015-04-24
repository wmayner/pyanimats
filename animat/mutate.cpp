// mutate.cpp

#include <vector>

#include "./mutate.hpp"

#define randDouble ((double)rand() / (double)RAND_MAX)


vector<unsigned char> mutateGenome(vector<unsigned char> genome, double
        mutProb, double dupProb, double delProb, int minGenomeLength, int
        maxGenomeLength) {
    int size = genome.size();
    // Mutation
    for (int i = 0; i < size; i++) {
        if (randDouble < mutProb) {
            genome[i] = rand() & 255;
        }
    }
    // Duplication
    if ((randDouble < dupProb) && (size < maxGenomeLength)) {
        int width = MIN_DUP_DEL_LENGTH + rand() & MAX_DUP_DEL_LENGTH;
        int start = rand() % (size - width);
        int insert = rand() % size;
        vector<unsigned char> buffer;
        buffer.clear();
        buffer.insert(buffer.begin(), genome.begin() + start, genome.begin() +
                start + width);
        genome.insert(genome.begin() + insert, buffer.begin(), buffer.end());
    }
    // Deletion
    if ((randDouble < delProb) && (size > minGenomeLength)) {
        int width = MIN_DUP_DEL_LENGTH + rand() & MAX_DUP_DEL_LENGTH;
        int start = rand() % (size - width);
        genome.erase(genome.begin() + start, genome.begin() + start + width);
    }
    return genome
}
