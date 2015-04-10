// Game.cpp

#include <vector>
#include <bitset>

#include "Game.hpp"

// TODO(wmayner) figure out and document cruel black magic voodoo sorcery
#define KISSRND (                                                        \
    ((((rndZ = 36969 * (rndZ & 65535) + (rndZ >> 16)) << 16) +           \
      (rndW = 18000 * (rndW & 65535) + (rndW >> 16)) )                   \
     ^(rndY = 69069 * rndY + 1234567)) +                                 \
    (rndX ^= (rndX << 17), rndX ^= (rndX >> 13), rndX ^= (rndX << 5))    \
)
#define INTABS(number) (((((0x80) << ((sizeof(int) - 1) << 3)) & number) \
            ? (~number) + 1 : number))

#define randDouble ((double)rand() / (double)RAND_MAX)

int rndX, rndY, rndZ, rndW;

int randInt(int i) {
    return rand() % i;
}

Game::Game(char* filename) {
    FILE *f = fopen(filename, "r+w");
    int i;
    patterns.clear();
    while (!feof(f)) {
        // TODO(wmayner) use CSV format
        fscanf(f, "%i  ", &i);
        patterns.push_back(bitset<WORLD_WIDTH>(i));
    }
    fclose(f);
}

Game::~Game() {}

double Game::agentDependentRandDouble(void) {
    int A = KISSRND;
    return (double)((INTABS(A)) & 65535) / (double)65535;
}

int Game::agentDependentRandInt(void) {
    int A = KISSRND;
    return (INTABS(A));
}

void Game::applyNoise(Agent *agent, double sensorNoise) {
    // Larissa: If I don't have noise in evaluation, then I can just use random
    // numbers always
    // if (agentDependentRandDouble() < sensorNoise) {
    if (randDouble < sensorNoise) {
        agent->states[0] = !agent->states[0];
    }
    // if (agentDependentRandDouble() < sensorNoise)
    if (randDouble < sensorNoise) {
        agent->states[1] = !agent->states[1];
    }
}

/**
 * Executes a game, updates the agent's fitness accordingly, and returns a
 * vector of the agent's state transitions over the course of the game.
 */
vector< vector<int> > Game::executeGame(Agent* agent, double sensorNoise, int
        repeat) {
    bitset<WORLD_WIDTH> world_state, old_world_state;

    vector< bitset<WORLD_WIDTH> > world;
    world.clear();
    world.resize(WORLD_HEIGHT);

    // Permutation that redirects agent's sensors. Defaults to doing nothing
    // (identity permutation).
    vector<int> worldTransform;
    worldTransform.resize(WORLD_WIDTH);
    for (int i = 0; i < WORLD_WIDTH; i++) worldTransform[i] = i;

    int initAgentPos, agentPos, past_state, current_state;
    int patternIndex, direction, timestep;
    int action;

    // This holds the state transitions over the agent's lifetime
    vector< vector<int> > stateTransitions;
    stateTransitions.clear();
    stateTransitions.resize(2);

    // Make random seeds unique from one another by including index
    rndW = agent->ID + repeat;
    rndX = ~(agent->ID + repeat);
    rndY = (agent->ID + repeat)^0b01010101010101010101010101010101;
    rndZ = (agent->ID + repeat)^0b10101010101010101010101010101010;

    agent->fitness = 1.0;
    agent->correct = agent->incorrect = 0;

    bool hit;

    // Record the number of correct outcomes for each different type of block
    agent->numCorrectByPattern.resize(patterns.size());
    for (int i = 0; i < agent->numCorrectByPattern.size(); i++) {
        agent->numCorrectByPattern[i] = 0;
    }

    // Block patterns
    for (patternIndex = 0; patternIndex < patterns.size(); patternIndex++) {
        // Directions (left/right)
        for (direction = -1; direction < 2; direction += 2) {
            // Agent starting position
            for (initAgentPos = 0; initAgentPos < WORLD_WIDTH; initAgentPos++) {
                agentPos = initAgentPos;

                // Larissa: Change environment after 30,000 Gen, if patterns is
                // 1 7 15 3 it changes from 2 blocks with 1 7 to 4 blocks with
                // 1 7 15 3

                // TODO(wmayner) add logic outside of Game to change the
                // patterns mid-evolution

                agent->resetBrain();

                // Generate world
                world.resize(WORLD_HEIGHT);
                world_state = patterns[patternIndex];

                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    world[timestep] = world_state;
                    old_world_state = world_state;
                    // Move the block
                    if (direction == -1) {
                        // Left
                        world_state = old_world_state >> 1;
                        world_state[WORLD_WIDTH - 1] = old_world_state[0];
                    } else {
                        // Right
                        world_state = old_world_state << 1;
                        world_state[0] = old_world_state[WORLD_WIDTH - 1];
                    }
                }

                if (SCRAMBLE_WORLD) {
                    // Scramble time
                    random_shuffle(world.begin(), world.end(), randInt);
                    // Scramble space (what animat sees will be determined by
                    // the transform)
                    random_shuffle(worldTransform.begin(),
                            worldTransform.end(), randInt);
                }

                // World loop
                for (timestep = 0; timestep < WORLD_HEIGHT; timestep++) {
                    world_state = world[timestep];

                    // Activate sensors if block is in line of sight
                    // TODO(wmayner) parametrize sensor location on agent body
                    agent->states[0] = world_state[worldTransform[agentPos]];
                    agent->states[1] = world_state[worldTransform[agentPos + 2]];

                    // TODO(wmayner) parameterize changing sensors mid-evolution
                    // Larissa: Set to 0 to evolve agents with just one sensor

                    applyNoise(agent, sensorNoise);

                    // Set motors to 0 to prevent them from influencing next
                    // animat state
                    // TODO(wmayner) parametrize motor node indices
                    agent->states[6] = 0;
                    agent->states[7] = 0;

                    past_state = 0;
                    for (int n = 0; n < NUM_NODES; n++) {
                        // Set the nth bit to the nth node's state
                        past_state |= (agent->states[n] & 1) << n;
                    }
                    stateTransitions[0].push_back(past_state);

                    agent->updateStates();

                    current_state = 0;
                    for (int n = 0; n < NUM_NODES; n++) {
                        // Set the nth bit to the nth node's state
                        current_state |= (agent->states[n] & 1) << n;
                    }
                    stateTransitions[1].push_back(current_state);

                    // TODO(wmayner) parameterize this
                    // Larissa: limit to one motor
                    // agent->states[7]=0;
                    // if (agent->born < nowUpdate) {
                    //     agent->states[7] = 0;
                    // }
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
                            agentPos = (agentPos + 1) % WORLD_WIDTH;
                            break;
                        // Right motor on
                        case 2:
                            // Move left
                            agentPos = (agentPos - 1) % WORLD_WIDTH;
                            break;
                    }
                }

                // Check for hit
                hit = false;
                // TODO(wmayner) un-hardcode agent body size
                for (int i = 0; i < 3; i++) {
                    if (world_state[(agentPos + i) % WORLD_WIDTH] == 1) {
                        hit = true;
                    }
                }

                // Update fitness
                // TODO(wmayner) Make the alternating catch/avoid stuff
                // explicit and read it from the file
                if ((patternIndex & 1) == 0) {
                    if (hit) {
                        agent->correct++;
                        agent->fitness *= FITNESS_MULTIPLIER;
                        agent->numCorrectByPattern[patternIndex]++;
                    } else {
                        agent->fitness /= FITNESS_MULTIPLIER;
                        agent->incorrect++;
                    }
                } else {
                    if (hit) {
                        agent->incorrect++;
                        agent->fitness /= FITNESS_MULTIPLIER;
                    } else {
                        agent->correct++;
                        agent->fitness *= FITNESS_MULTIPLIER;
                        agent->numCorrectByPattern[patternIndex]++;
                    }
                }
            }  // Agent starting position
        }  // Directions
    }  // Block patterns
    return stateTransitions;
}  // executeGame
