// Game.h

#ifndef ANIMAT_GAME_H_
#define ANIMAT_GAME_H_

#include <vector>

#include "./Agent.hpp"
#include "./constants.hpp"
#include "./rng.hpp"

using std::vector;

vector<int> executeGame(std::vector<unsigned char> &allAnimatStates,
        std::vector<int> &allWorldStates, vector<int> &allAnimatPositions,
        vector<int> &trialResults, Agent* agent, vector<int> hit_multipliers,
        vector<int> patterns, int worldWidth, int worldHeight, bool
        scrambleWorld);

#endif  // ANIMAT_GAME_H_
