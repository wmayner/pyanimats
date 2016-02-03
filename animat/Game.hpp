// Game.h

#ifndef SRC_GAME_H_
#define SRC_GAME_H_

#include <vector>

#include "./constants.hpp"
#include "./Agent.hpp"

using std::vector;

void executeGame(std::vector<unsigned char> &allAnimatStates, std::vector<int>
        &allWorldStates, vector<int> &allAnimatPositions, vector<int>
        &trialResults, Agent* agent, vector<int> hit_multipliers,
        vector<int> patterns, int worldWidth, int worldHeight,
        bool scrambleWorld);

#endif  // SRC_GAME_H_
