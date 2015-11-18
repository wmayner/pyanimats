Usage
==========

help - `python pyanimats.py -h`
basic usage - `pyanimats.py <output_dir> <tasks.yml> [options]`

Installation
--------------
1. Make virtualenv in Python 3
    * Mac

            `mkvirtualenv -p which python3 PyAnimats`

    * Linux

            `mkvirtualenv -p /usr/bin/python3 PyAnimats`

1. `pip install -r requirements.txt`

    (make sure gcc is installed because you need gfortran compiler for scipy: brew install gcc)

1. Rebuild each time any c++ changes

    `python setup.py build_ext --inplace`



Setup of a Task
-----------------

1. Prepare a task as a yaml file, example:
    ```yaml
    # tasks/1-3-1-3.yml
    # PyAnimat goals and tasks.
    # 1 means catch the block, -1 means avoid it.
    ---
    # 1-3-1-3 task.
    - [ 1, '1000000000000000']
    - [-1, '1110000000000000']
    - [ 1, '1000000000000000']
    - [-1, '1110000000000000']
    ```
    
1. Remove old build, and then rebuild it each time any c++ files change, (eg `animats/constants.hpp`)
    
    `rm -r build && python setup.py build_ext --inplace`

1. Run it with 30,000 generations, and sample 600 individuals from the evolution
   
    `python pyanimats.py ./raw_results/0.0.21/initial_tests/seed-0 ./tasks/1-3-1-3.yml -n 30000 -i 600`


Outputs
-------

In your selected output dir, will appear the following files:

* `config.json`

  The configuration that this game was run with, including for example: `NUM_MOTORS, TASKS, NUM_NODES, MUTATION_PROB`, etc.

* `hof.pkl` - "Hall of Fame"

    Individual's animat

* `lineages.pkl`


* `logbook.pkl`


* `metadata.json`

    Contains `elapsed` - seconds elapsed, and `version` - the version of the code


Josh's Questions
---------


main entry points
-------------------

* `analyze.py`
* `plot.py`
* `pyanimats.py`
* `make_pkl_for_pyphi.py`



* Main Inputs

  `config.py` - 
  `configure.py` - 
  `constants.py` - 
  `params.yml` - 
  `tasks/*.yml` - 
  `animat/constants.hpp` - 

* Main Outputs

  `config.json` - 
  `metadata.json` - 
  `hof.pkl` - 
  `lineages.pkl` - 
  `logbook.pkl` - 




game related
-------------

* `Game.cpp`

* `HMM.cpp`


agent related
--------------

* `Agent.cpp`

* `animat.cpp`

* `Individual.py`

   
