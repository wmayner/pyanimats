// Game.h

#ifndef _PyAnimats_Game_H
#define _PyAnimats_Game_H

#include <vector>

#include "./AbstractAgent.hpp"
#include "./constants.hpp"
#include "./rng.hpp"

using std::vector;

vector<int> executeGame(std::vector<unsigned char> &allAnimatStates,
        std::vector<int> &allWorldStates, vector<int> &allAnimatPositions,
        vector<int> &trialResults, AbstractAgent* agent,
        vector<int> hit_multipliers, vector<int> patterns, int worldWidth,
        int worldHeight, bool scrambleWorld);

#endif  // _PyAnimats_Game_H
