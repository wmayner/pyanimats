
"""
Export PyAnimats to Visual Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    export.py <output.json> evolution <evolution.json>
    export.py <output.json> game <evolution.json>

Arguments:
    <output.json>               File where the visual interface JSON should be
                                stored
    evolution <evolution.json>  Export an evolution to the JSON format used by
                                the evolution visual interface
    game <evolution.json>       Export the fittest animat in the evolution to
                                the JSON format used by the game visual
                                interface
"""

import json

import numpy as np
import pyphi
from pyphi.convert import loli_index2state as i2s
from docopt import docopt

from pyanimats import evolve, fitness_functions, serialize, utils
from pyanimats.__about__ import __version__


def num_concepts(ind, state):
    mc = pyphi.compute.main_complex(ind.network, state)
    return len(mc.unpartitioned_constellation)

def phi(ind, state):
    mc = pyphi.compute.main_complex(ind.network, state)
    return mc.phi


get_phi = fitness_functions.avg_over_visited_states()(phi)
get_num_concepts = fitness_functions.avg_over_visited_states()(num_concepts)


# TODO: average over all visited states, not unique visited states?
def convert_evolution_to_json(evolution):
    """Convert an an evolution to the json format used by `animanimate`. The
     JSON produced by this funciton is the input for the evolution tab."""
    return serialize.serializable([
        {
            'fitness': animat.fitness,
            'phi': get_phi(animat),
            'numConcepts': get_num_concepts(animat),
            'cm': animat.cm,
            'mechanisms': animat.mechanisms(separate_on_off=True),
            'config': get_config(animat),
        }
        for animat in evolution.lineage
    ])


def get_config(animat):
    return {
        'NUM_NODES': animat.num_nodes,
        'NUM_SENSORS': animat.num_sensors,
        'SENSOR_INDICES': animat.sensor_indices,
        'MOTOR_INDICES': animat.motor_indices,
        'HIDDEN_INDICES': animat.hidden_indices,
        'SENSOR_LOCATIONS': animat.sensor_locations,
        'BODY_LENGTH': animat.body_length,
        'SEED': animat.rng_seed,
        'FITNESS_FUNCTION': animat.fitness_function,
        'WORLD_HEIGHT': animat.world_height,
        'WORLD_WIDTH': animat.world_width,
    }


def get_phi_data(animat, game):
    """Calculate the IIT properties of the given animat for every state.

    The data function must take and individual and a state.
    """
    # # TODO: handle multiple fitness functions
    # assert len(animat.fitness_function) == 1
    # ff = animat.fitness_function[0]
    #
    # # Get the function that returns the data (before condensing it into a
    # # simple fitness value).
    # data_func = fitness_functions.metadata[ff]['data_function']
    # if data_func is None:
    #     return None

    # Get the data for every state.
    return {state: fitness_functions.main_complex(animat, state).to_json()
            for state in map(tuple, utils.unique_rows(game.animat_states))}


def convert_animat_to_game_json(animat, scrambled=False):
    """Convert an animat to the json format used by the game tab of
    `animanimate`."""

    # Play the game
    game = animat.play_game(scrambled=scrambled)

    phi_data = get_phi_data(animat, game)

    world_width = animat.world_width
    # Convert world states from the integer encoding to explicit arrays.
    world_states = np.array(
        list(map(lambda i: i2s(i, world_width),
                 game.world_states.flatten().tolist()))).reshape(
                     game.world_states.shape + (world_width,))

    return serialize.serializable({
        'config': get_config(animat),
        'generation': animat.gen,
        'fitness': animat.fitness,
        'correct': animat.correct,
        'incorrect': animat.incorrect,
        'cm': animat.cm,
        'mechanisms': animat.mechanisms(separate_on_off=True),
        'notes': None,
        'trials': [
            {
                'num': trialnum,
                # First result bit is whether the block was caught.
                'catch': bool(game.trial_results[trialnum] & 1),
                # Second result bit is whether the animat was correct.
                'correct': bool((game.trial_results[trialnum] >> 1) & 1),
                'timesteps': [
                    {
                        'num': t,
                        'world': world_state,
                        'animat': animat_state,
                        'pos': animat_position,
                        'phidata': phi_data[tuple(animat_state)]
                    }
                    for t, (world_state, animat_state, animat_position)
                        in enumerate(zip(
                            world_states[trialnum],
                            game.animat_states[trialnum],
                            game.animat_positions[trialnum]
                        ))
                ],
            } for trialnum in range(game.animat_states.shape[0])
        ],
    })


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    with open(args['<evolution.json>']) as f:
        data = json.load(f)

    evolution = evolve.from_json(data)

    if args['evolution']:
        output = convert_evolution_to_json(evolution)

    elif args['game']:
        fittest = evolution.lineage[0]
        output = convert_animat_to_game_json(fittest)

    with open(args['<output.json>'], 'w') as f:
        json.dump(output, f, indent=4)
