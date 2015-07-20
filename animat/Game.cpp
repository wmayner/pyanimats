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
 * vector of the agent's state transitions over the course of the game
 */
void executeGame(vector<unsigned char> &allAnimatStates, vector<int>
        &allWorldStates, vector<int> &allAnimatPositions, Agent* agent,
        vector<int> hitMultipliers, vector<int> patterns, bool scrambleWorld) {
    vector<int> world;
    world.clear();
    world.resize(WORLD_HEIGHT);

    // Permutation that redirects agent's sensors. Defaults to doing nothing
    // (identity permutation)
    vector<int> worldTransform;
    worldTransform.resize(WORLD_WIDTH);
    for (int i = 0; i < WORLD_WIDTH; i++) worldTransform[i] = i;

    int initAgentPos, agentPos;
    int patternIndex, direction, timestep;
    int action;

    int allAnimatStatesIndex = 0;
    int allWorldStatesIndex = 0;
    int allAnimatPositionsIndex = 0;
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
                int worldState = patterns[patternIndex];

                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    world[timestep] = worldState;
                    // Move the block
                    if (direction == -1) {
                        // Left
                        worldState = ((worldState >> 1) & 65535) +
                            ((worldState & 1) << (WORLD_WIDTH - 1));
                    } else {
                        // Right
                        worldState = ((worldState << 1) & 65535) +
                            ((worldState >> (WORLD_WIDTH - 1)) & 1);
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

                #ifdef _DEBUG
                    printf("\n\n-------------------------");
                    printf("\n   Block pattern: %i", patterns[patternIndex]);
                    printf("\n       Direction: %i", direction);
                    printf("\nInitial position: %i", initAgentPos);
                    printf("\n\n");
                #endif

                // World loop
                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    worldState = world[timestep];
                    // Record the world state
                    allWorldStates[allWorldStatesIndex++] = worldState;
                    // Record agent position
                    allAnimatPositions[allAnimatPositionsIndex++] = agentPos;

                    // Activate sensors if block is in line of sight
                    // TODO(wmayner) parametrize sensor location on agent body
                    if (NUM_SENSORS == 2) {
                        agent->states[0] = (worldState >>
                                worldTransform[agentPos]) & 1;
                        agent->states[1] = (worldState >>
                                worldTransform[wrap(agentPos + 2)]) & 1;
                    }
                    if (NUM_SENSORS == 3) {
                        for (int i = 0; i < 3; i++)
                            agent->states[i] = (worldState >>
                                    worldTransform[wrap(agentPos + i)]) & 1;
                    }

                    #ifdef _DEBUG
                        // Print the world
                        for (int i = 0; i < (int)WORLD_WIDTH; i++)
                            printf("%i", (worldState >> i) & 1);
                        printf("\n");

                        // Print the animat
                        bool space;
                        for (int i = 0; i < WORLD_WIDTH; i++) {
                            space = true;
                            for (int k = 0; k < 3; k++)
                                if (wrap(agentPos + k) == i) {
                                    if (NUM_SENSORS == 3) {
                                        printf("%i", agent->states[k]);
                                    } else {
                                        if (k == 0)
                                            printf("%i", agent->states[0]);
                                        if (k == 1)
                                            printf("-");
                                        if (k == 2)
                                            printf("%i", agent->states[1]);
                                    }
                                    space = false;
                                }
                            if (space) {
                                printf(" ");
                            }
                        }
                        printf("\n");
                    #endif

                    // TODO(wmayner) parameterize changing sensors mid-evolution
                    // Larissa: Set to 0 to evolve agents with just one sensor

                    // Record state of sensors
                    for (int n = 0; n < NUM_SENSORS; n++)
                        allAnimatStates[allAnimatStatesIndex++] = agent->states[n];

                    agent->updateStates();

                    // Record state of hidden units and motors after updating animat
                    for (int n = NUM_SENSORS; n < NUM_NODES; n++) {
                        allAnimatStates[allAnimatStatesIndex++] = agent->states[n];
                    }

                    // Update hitcount if this is the last timestep
                    if (timestep == WORLD_HEIGHT - 1) {
                        int hit = 0;
                        // TODO(wmayner) un-hardcode agent body size
                        for (int i = 0; i < 3; i++) {
                            if (((worldState >> (wrap(agentPos + i))) & 1)
                                    == 1)
                                hit = 1;
                        }
                        #ifdef _DEBUG
                        printf("-----------------\n");
                        #endif
                        if (hitMultipliers[patternIndex] > 0) {
                            if (hit == 1) {
                                agent->correct++;
                                #ifdef _DEBUG
                                printf("CAUGHT (CORRECT!)");
                                #endif
                            }
                            else {
                                agent->incorrect++;
                                #ifdef _DEBUG
                                printf("AVOIDED (WRONG.)");
                                #endif
                            }
                        }
                        if (hitMultipliers[patternIndex] <= 0) {
                            if (hit == 0) {
                                agent->correct++;
                                #ifdef _DEBUG
                                printf("AVOIDED (CORRECT!)");
                                #endif
                            }
                            else {
                                agent->incorrect++;
                                #ifdef _DEBUG
                                printf("CAUGHT (WRONG.)");
                                #endif
                            }
                        }
                        // Break out of the world loop, since the animat's
                        // subsequent movement doesn't count
                        break;
                    }

                    // TODO(wmayner) switch motors and cases to be less
                    // confusing
                    action = agent->states[6] + (agent->states[7] << 1);

                    // Move agent
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
                } // End world loop
            }  // Agent starting position
        }  // Directions
    }  // Block patterns
}  // executeGame
