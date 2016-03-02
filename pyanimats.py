#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyanimats.py

"""
PyAnimats
~~~~~~~~~
Evolve animats.

Usage:
    pyanimats.py run <path/to/experiment.yml> <path/to/output_file> [options]
    pyanimats.py resume <path/to/checkpoint.pkl> <path/to/output_file>
    pyanimats.py -h | --help
    pyanimats.py -v | --version
    pyanimats.py --list

Command-line options override the parameters given in the experiment file.

Options:
    --list                    List available fitness functions
    -h --help                 Show this
    -v --version              Show version
    -r --rng-seed=INT         Random number generator seed
    -t --snapshot=INT         Snapshot interval (minutes)
    -s --status-interval=INT  Status-printing interval (generations)
    -o --min-snapshots=INT    Minimum number of snapshots to take
    -l --log-interval=INT     Logbook recording interval (generations)
    -i --num-samples=INT      Number of animats to sample from evolution
    -f --fitness=FUNC         Fitness function
    -n --num-gen=NGEN         Number of generations to simulate
    -p --pop-size=INT         Population size
    -g --init-genome=PATH     Path to a lineage file for an intial genome
    -j --jumpstart=INT        Begin with this many start codons
    -a --all-lineages         Save lineages of entire final population
    -e --num-sensors=INT      The number of sensors in an animat
    -d --num-hidden=INT       The number of hidden units in an animat
    -t --num-motors=INT       The number of motors in an animat
    -W --world-width=INT      The width of the animats' environment
    -H --world-height=INT     The height of the animats' environment
    -m --mut-prob=FLOAT       Point mutation probability
       --dup-prob=FLOAT       Duplication probability
       --del-prob=FLOAT       Deletion probability
       --min-dup-del=INT      Minimum length of duplicated/deleted genome part
       --max-dup-del=INT      Maximum length of duplicated/deleted genome part
       --min-length=INT       Minimum genome length
       --max-length=INT       Maximum genome length
       --profile=PATH         Profile performance and store results at PATH
    -F --force                Overwrite the output file.
    -c --checkpoint=PATH      Save to this checkpoint file (defaults to
                              `checkpoint.pkl` in the output directory).
"""

import cProfile
import os
import random

from deap import base, tools
from docopt import docopt

import c_animat
import fitness_functions
import utils
from __about__ import __version__
from animat import Animat
from evolve import Evolution, load_checkpoint
from experiment import Experiment

# Map CLI options to experiment parameter names and types.
cli_opt_to_param = {
    '--rng-seed':        ('rng_seed', int),
    '--snapshot':        ('snapshot_frequency', int),
    '--status-interval': ('status_interval', int),
    '--min-snapshots':   ('min_snapshots', int),
    '--log-interval':    ('log_interval', int),
    '--num-samples':     ('num_samples', int),
    '--fitness':         ('fitness_function', str),
    '--num-gen':         ('ngen', int),
    '--pop-size':        ('popsize', int),
    '--init-genome':     ('init_genome', str),
    '--jumpstart':       ('init_start_codons', int),
    '--num-sensors':     ('num_sensors', int),
    '--num-hidden':      ('num_hidden', int),
    '--num-motors':      ('num_motors', int),
    '--world-width':     ('world_width', int),
    '--world-height':    ('world_height', int),
    '--mut-prob':        ('mutation_prob', float),
    '--dup-prob':        ('duplication_prob', float),
    '--del-prob':        ('deletion_prob', float),
    '--min-dup-del':     ('min_dup_del_width', int),
    '--max-dup-del':     ('min_dup_del_width', int),
    '--min-length':      ('min_genome_length', int),
    '--max-length':      ('max_genome_length', int),
}


def main(arguments):

    # Handle arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO make this an option for -h?
    # Print available fitness functions and their descriptions.
    if arguments['--list']:
        fitness_functions.print_functions()
        return 0

    # Final output will be written here.
    OUTPUT_FILE = arguments['<path/to/output_file>']
    # Don't overwrite the output file or pwithout permission.
    if not arguments['--force'] and os.path.exists(OUTPUT_FILE):
        raise FileExistsError(
            'a file named `{}` already exists; not overwriting without the '
            '`--force` option.'.format(OUTPUT_FILE))
    # Checkpoints will be written here.
    CHECKPOINT_FILE = (arguments['--checkpoint'] or
                       os.path.join(os.path.dirname(OUTPUT_FILE),
                                    'checkpoint.pkl'))

    # Setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize the DEAP toolbox.
    toolbox = base.Toolbox()

    # Either load from a checkpoint or start a new evolution.
    if arguments['resume']:
        # Load the checkpoint.
        checkpoint = load_checkpoint(CHECKPOINT_FILE)
        # Convert the loaded Phylogeny back to a normal list.
        population = checkpoint['population']
        experiment = checkpoint['experiment']
        logbook = checkpoint['logbook']
        python_rng_state = checkpoint['python_rng_state']
        c_rng_state = checkpoint['c_rng_state']
        start_generation = checkpoint['generation']
        print('Loaded checkpoint from `{}`.'.format(CHECKPOINT_FILE))
        print('Resuming evolution from generation '
              '{}...'.format(start_generation))
    else:
        # Load the experiment object, overriding if necessary with CLI options.
        cli_overrides = {param[0]: param[1](arguments[opt])
                         for opt, param in cli_opt_to_param.items()
                         if arguments[opt] is not None}
        experiment = Experiment(filepath=arguments['<path/to/experiment.yml>'],
                                override=cli_overrides)
        # Register the various genetic algorithm components to the toolbox.
        toolbox.register('animat', Animat, experiment, experiment.init_genome)
        toolbox.register('population', tools.initRepeat, list, toolbox.animat)
        # Initialize logbooks and hall of fame.
        logbook = tools.Logbook()
        # Create initial population.
        population = toolbox.population(n=experiment.popsize)
        # Seed the random number generators.
        random.seed(experiment.rng_seed)
        c_animat.seed(experiment.rng_seed)
        # Get their states to pass to the evolution.
        python_rng_state = random.getstate()
        c_rng_state = c_animat.get_rng_state()
        start_generation = 0
        print('\nSimulating {} generations...'.format(experiment.ngen))

    # Initialize the simulation.
    evolution = Evolution(experiment, population, logbook, python_rng_state,
                          c_rng_state, start_generation)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    PROFILE_FILEPATH = arguments['--profile']
    if PROFILE_FILEPATH:
        utils.ensure_exists(os.path.dirname(PROFILE_FILEPATH))
        pr = cProfile.Profile()
        pr.enable()

    # Run it!
    evolution.run(CHECKPOINT_FILE)

    if PROFILE_FILEPATH:
        pr.disable()
        pr.dump_stats(PROFILE_FILEPATH)

    print('\nSimulated {} generations in {}.'.format(
        evolution.generation, utils.compress(evolution.elapsed)))
    print('\nSaving data to `{}`... '.format(OUTPUT_FILE), end='')

    # Get the evolution results.
    output = evolution.to_json(all_lineages=arguments['--all-lineages'])
    # Ensure output directory exists and write to disk.
    utils.ensure_exists(os.path.dirname(OUTPUT_FILE))
    with open(OUTPUT_FILE, 'w') as f:
        utils.dump(output, f)

    print('done.')


if __name__ == '__main__':
    # Get command-line arguments from docopt.
    arguments = docopt(__doc__, version=__version__)
    main(arguments)
