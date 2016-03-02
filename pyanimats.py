#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyanimats.py

"""
PyAnimats
~~~~~~~~~
Evolve animats.

Usage:
    pyanimats.py run <path/to/experiment.yml> <path/to/output_file> [options]
    pyanimats.py resume <path/to/checkpoint.pkl> <path/to/output_file> [options]
    pyanimats.py -h | --help
    pyanimats.py -v | --version
    pyanimats.py --list

Command-line options override the parameters given in the experiment file.

Options:
    --list                     List available fitness functions
    -h --help                  Show this
    -v --version               Show version
    -F --force                 Overwrite the output file.
    -r --rng-seed=INT          Random number generator seed
    -c --checkpoint=INT        Checkpoint interval (minutes)
    -C --checkpoint-file=PATH  Save to this checkpoint file (defaults to
                               `checkpoint.pkl` in the output directory)
    -s --status-interval=INT   Status-printing interval (generations)
    -b --logbook-interval=INT  Logbook recording interval (generations)
    -o --output-samples=INT    Number of animats to sample from evolution
    -f --fitness=FUNC          Fitness function
    -n --num-gen=NGEN          Number of generations to simulate
    -p --pop-size=INT          Population size
    -g --init-genome=PATH      Path to a lineage file for an intial genome
    -j --jumpstart=INT         Begin with this many start codons
    -e --num-sensors=INT       The number of sensors in an animat
    -i --num-hidden=INT        The number of hidden units in an animat
    -t --num-motors=INT        The number of motors in an animat
    -W --world-width=INT       The width of the animats' environment
    -H --world-height=INT      The height of the animats' environment
    -m --mut-prob=FLOAT        Point mutation probability
    -U --dup-prob=FLOAT        Duplication probability
    -E --del-prob=FLOAT        Deletion probability
    -d --min-dup-del=INT       Minimum length of duplicated/deleted genome part
    -D --max-dup-del=INT       Maximum length of duplicated/deleted genome part
    -l --min-length=INT        Minimum genome length
    -L --max-length=INT        Maximum genome length
    -a --all-lineages          Save lineages of entire final population
    -P --profile=PATH          Profile performance and store results at PATH
"""

import cProfile
import os
import pickle

from docopt import docopt

import fitness_functions
import utils
from __about__ import __version__
from evolve import Evolution
from experiment import Experiment

# Map CLI options to experiment parameter names and types.
cli_opt_to_param = {
    '--rng-seed':         ('rng_seed', int),
    '--checkpoint':       ('checkpoint_interval', int),
    '--status-interval':  ('status_interval', int),
    '--logbook-interval': ('logbook_interval', int),
    '--output-samples':   ('output_samples', int),
    '--fitness':          ('fitness_function', str),
    '--num-gen':          ('ngen', int),
    '--pop-size':         ('popsize', int),
    '--init-genome':      ('init_genome', str),
    '--jumpstart':        ('init_start_codons', int),
    '--num-sensors':      ('num_sensors', int),
    '--num-hidden':       ('num_hidden', int),
    '--num-motors':       ('num_motors', int),
    '--world-width':      ('world_width', int),
    '--world-height':     ('world_height', int),
    '--mut-prob':         ('mutation_prob', float),
    '--dup-prob':         ('duplication_prob', float),
    '--del-prob':         ('deletion_prob', float),
    '--min-dup-del':      ('min_dup_del_width', int),
    '--max-dup-del':      ('max_dup_del_width', int),
    '--min-length':       ('min_genome_length', int),
    '--max-length':       ('max_genome_length', int),
}


def main(arguments):
    # TODO make this an option for -h?
    # Print available fitness functions and their descriptions.
    if arguments['--list']:
        fitness_functions.print_functions()
        return 0

    # Final output will be written here.
    OUTPUT_FILE = arguments['<path/to/output_file>']
    # Don't overwrite the output file or without permission.
    if not arguments['--force'] and os.path.exists(OUTPUT_FILE):
        raise FileExistsError(
            'a file named `{}` already exists; not overwriting without the '
            '`--force` option.'.format(OUTPUT_FILE))
    # Checkpoints will be written here.
    CHECKPOINT_FILE = (arguments['--checkpoint-file'] or
                       arguments['<path/to/checkpoint.pkl>'] or
                       os.path.join(os.path.dirname(OUTPUT_FILE),
                                    'checkpoint.pkl'))

    # Either load from a checkpoint or start a new evolution.
    if arguments['resume']:
        # Load the checkpoint.
        print('Loading checkpoint from `{}`... '.format(CHECKPOINT_FILE),
              end='', flush=True)
        with open(arguments['<path/to/checkpoint.pkl>'], 'rb') as f:
            evolution = pickle.load(f)
        print('done.')
        print('Resuming evolution from generation '
              '{}...\n'.format(evolution.generation))
    else:
        # Load the experiment object, overriding if necessary with CLI options.
        cli_overrides = {param[0]: param[1](arguments[opt])
                         for opt, param in cli_opt_to_param.items()
                         if arguments[opt] is not None}
        experiment = Experiment(filepath=arguments['<path/to/experiment.yml>'],
                                override=cli_overrides)
        # Initialize the simulation.
        evolution = Evolution(experiment)
        print('Simulating {} generations...'.format(experiment.ngen))

    PROFILE_FILEPATH = arguments['--profile']
    if PROFILE_FILEPATH:
        utils.ensure_exists(os.path.dirname(PROFILE_FILEPATH))
        print('\nProfiling enabled.')
        pr = cProfile.Profile()
        pr.enable()

    # Run it!
    evolution.run(CHECKPOINT_FILE)

    if PROFILE_FILEPATH:
        pr.disable()
        print('\nSaving profile to `{}`... '.format(PROFILE_FILEPATH),
              end='', flush=True)
        pr.dump_stats(PROFILE_FILEPATH)
        print('done.')

    print('\nSimulated {} generations in {}.'.format(
        evolution.generation, utils.compress(evolution.elapsed)))
    print('\nSaving output to `{}`... '.format(OUTPUT_FILE),
          end='', flush=True)

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
