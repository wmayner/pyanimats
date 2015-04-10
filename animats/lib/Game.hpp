// Game.h

#ifndef SRC_GAME_H_
#define SRC_GAME_H_

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "./constants.hpp"
#include "./Agent.hpp"

using std::vector;
using std::bitset;

class Game {
 public:
    vector< bitset<WORLD_WIDTH> > patterns;
    vector< vector<int> > executeGame(Agent* agent, double sensorNoise, int
            repeat = 0);
    explicit Game(char* filename);
    ~Game();

    void applyNoise(Agent *agent, double sensorNoise);
    double agentDependentRandDouble(void);
    int agentDependentRandInt(void);
    int nowUpdate;
};

#endif  // SRC_GAME_H_
