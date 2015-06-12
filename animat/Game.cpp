// Game.cpp

#include <algorithm>
#include <vector>

#include "./Game.hpp"

int randInt(int i) {
    return rand() % i;
}

int wrap(int i) {
    return i & (WORLD_WIDTH - 1);
}

/**
 * Executes a game, updates the agent's hit count accordingly, and returns a
 * vector of the agent's state transitions over the course of the game.
 */
void executeGame(vector<unsigned char> &stateTransitions, Agent* agent,
        vector<int> hitMultipliers, vector<int> patterns, bool scrambleWorld) {
    vector<int> world;
    world.clear();
    world.resize(WORLD_HEIGHT);

    // Permutation that redirects agent's sensors. Defaults to doing nothing
    // (identity permutation).
    vector<int> worldTransform;
    worldTransform.resize(WORLD_WIDTH);
    for (int i = 0; i < WORLD_WIDTH; i++) worldTransform[i] = i;

    int initAgentPos, agentPos;
    int patternIndex, direction, timestep;
    int action;

    int stateTransitionsIndex = 0;
    // Block patterns
    for (patternIndex = 0; patternIndex < (int)patterns.size(); patternIndex++) {
        // Directions (left/right)
        for (direction = -1; direction < 2; direction += 2) {
            // Agent starting position
            for (initAgentPos = 0; initAgentPos < WORLD_WIDTH; initAgentPos++) {
                // Set agent position
                agentPos = initAgentPos;

                // Larissa: Change environment after 30,000 Gen, if patterns is
                // 1 7 15 3 it changes from 2 blocks with 1 7 to 4 blocks with
                // 1 7 15 3

                // TODO(wmayner) add logic outside of Game to change the
                // patterns mid-evolution

                agent->resetState();

                // Generate world
                world.resize(WORLD_HEIGHT);
                int world_state = patterns[patternIndex];

                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    world[timestep] = world_state;
                    // Move the block
                    if (direction == -1) {
                        // Left
                        world_state = ((world_state >> 1) & 65535) +
                            ((world_state & 1) << (WORLD_WIDTH - 1));
                    } else {
                        // Right
                        world_state = ((world_state << 1) & 65535) +
                            ((world_state >> (WORLD_WIDTH - 1)) & 1);
                    }
                }

                if (scrambleWorld) {
                    // Scramble time
                    std::random_shuffle(world.begin(), world.end(), randInt);
                    // Scramble space (what animat sees will be determined by
                    // the transform)
                    std::random_shuffle(worldTransform.begin(),
                            worldTransform.end(), randInt);
                }

                // World loop
                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    world_state = world[timestep];

                    // Activate sensors if block is in line of sight
                    // TODO(wmayner) parametrize sensor location on agent body
                    agent->states[0] = (world_state >> worldTransform[agentPos]) & 1;
                    agent->states[1] = (world_state >> worldTransform[wrap(agentPos + 2)]) & 1;

                    // TODO(wmayner) parameterize changing sensors mid-evolution
                    // Larissa: Set to 0 to evolve agents with just one sensor

                    // Record state of sensors
                    for (int n = 0; n < NUM_SENSORS; n++)
                        stateTransitions[stateTransitionsIndex++] = agent->states[n];

                    agent->updateStates();

                    // Record state of hidden units and motors after updating animat
                    for (int n = NUM_SENSORS; n < NUM_NODES; n++) {
                        stateTransitions[stateTransitionsIndex++] = agent->states[n];
                    }

                    // TODO(wmayner) switch motors and cases to be less
                    // confusing
                    action = agent->states[6] + (agent->states[7] << 1);

                    // Move agent
                    // Larissa: this makes the agent stop moving:
                    // action = 0;
                    switch (action) {
                        // No motors on
                        case 0:
                            // Don't move
                            break;
                        // Both motors on
                        case 3:
                            // Don't move
                            break;
                        // Left motor on
                        case 1:
                            // Move right
                            agentPos = wrap(agentPos + 1);
                            break;
                        // Right motor on
                        case 2:
                            // Move left
                            agentPos = wrap(agentPos - 1);
                            break;
                    }
                }

                // Update hitcount
                int hit = 0;
                // TODO(wmayner) un-hardcode agent body size
                for (int i = 0; i < 3; i++) {
                    if (((world_state >> (wrap(agentPos + i))) & 1) == 1) {
                        hit = 1;
                    }
                }
                if (hitMultipliers[patternIndex] > 0) {
                    if (hit == 1) {
                        agent->correct++;
                    }
                    else {
                        agent->incorrect++;
                    }
                }
                if (hitMultipliers[patternIndex] <= 0) {
                    if (hit == 0) {
                        agent->correct++;
                    }
                    else {
                        agent->incorrect++;
                    }
                }
            }  // Agent starting position
        }  // Directions
    }  // Block patterns
}  // executeGame
