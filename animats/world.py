#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# animat.py

from bitarray import bitarray

from .constants import NUM_NODES, BODY_SIZE, NUM_SENSORS, SENSOR_LOCATIONS

LEFT = -1
RIGHT = 1

M1_INDEX = NUM_NODES - 2
M2_INDEX = NUM_NODES - 1

WIDTH = 16
HEIGHT = 36

INITIAL_ANIMAT_POSITIONS = range(WIDTH)

WIDTH_LESS_ONE = WIDTH - 1
def get_next_state(past_state, direction):
    """The state is an `bitarray`. Bits that are on represent cells in the
    world line that are occupied by a block cell."""
    next_state = bitarray(WIDTH)
    if direction == LEFT:
        next_state[0: WIDTH_LESS_ONE] = past_state[1:]
        next_state[WIDTH_LESS_ONE - 1] = past_state[0]
    elif direction == RIGHT:
        next_state[1:] = past_state[0: WIDTH_LESS_ONE]
        next_state[0] = past_state[WIDTH_LESS_ONE]
    return next_state


class World:

    """A with varying patterns of blocks in which the animat must either catch
    or avoid the block."""

    def __init__(self, animat, tasks):
        self.animat = animat
        self.tasks = tasks

        self.animat.fitness = 1.0

    def run(self):
        history = {}
        for goal, init_world_state in self.tasks:
            history[init_world_state] = {}
            for direction in (LEFT, RIGHT):
                history[init_world_state][direction] = {}

                # Generate world.
                world_states = [bitarray(init_world_state)]
                for i in range(1, HEIGHT):
                    # TODO(wmayner) make this faster by eliminating need to
                    # cast to string and slice
                    world_states.append(get_next_state(world_states[i - 1],
                                                       direction))

                # Play game for each animat starting position.
                for init_animat_pos in INITIAL_ANIMAT_POSITIONS:
                    animat_pos = init_animat_pos
                    animat_states = []
                    for timestep in range(HEIGHT):
                        world = world_states[timestep]
                        # Activate animat sensors based on world.
                        # NOTE: Sensors must always be the first nodes.
                        for sensor in range(NUM_SENSORS):
                            self.animat.state[sensor] = \
                                world[(animat_pos +
                                       SENSOR_LOCATIONS[sensor]) % WIDTH]

                        # Record animat state.
                        animat_states.append(self.animat.state)

                        # Move animat.
                        self.animat.update_state()
                        if (self.animat.state[M1_INDEX] and not
                                self.animat.state[M2_INDEX]):
                            animat_pos = animat_pos - 1 % WIDTH
                        elif (not self.animat.state[M1_INDEX] and
                              self.animat.state[M2_INDEX]):
                            animat_pos = animat_pos + 1 % WIDTH

                    # Record game history.
                    history[init_world_state][direction][init_animat_pos] = \
                        animat_states

                    # Determine if animat caught the falling block.
                    catch = False
                    for i in range(BODY_SIZE):
                        if world[(animat_pos + i) % WIDTH]:
                            catch = True

                    # Record correct/incorrect catches.
                    self.animat.correct += int(goal == catch)
                    self.animat.incorrect += int(goal != catch)

        return history

    def __repr__(self):
        return 'World(' + repr(self.animat) + ')'

    def __str__(self):
        return repr(self)
