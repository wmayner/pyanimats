// Game.h

#ifndef SRC_GAME_H_
#define SRC_GAME_H_

#include <vector>

#include "./constants.hpp"
#include "./Agent.hpp"

using std::vector;
using std::bitset;

vector< vector<int> > executeGame(Agent* agent, vector<int> patterns, bool
        scrambleWorld);

#endif  // SRC_GAME_H_
