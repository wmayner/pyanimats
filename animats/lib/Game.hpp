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

vector< vector<int> > executeGame(Agent* agent);

#endif  // SRC_GAME_H_
