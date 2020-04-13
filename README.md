# pyanimats

## Installation

This package must be installed with [`conda`](https://docs.conda.io/en/latest/miniconda.html).

1. Create the environment from the `environment.yml` file: `conda env create --name=<name_of_your_environment> --file=environment.yml`
2. Activate the environment: `conda activate <name_of_your_environment>`
3. Build the C++ extensions by running `make`.
4. Check that you can run the example simulation: from within the top-level directory of the repository, run `python -m pyanimats <output_directory> run experiments/example.yml`


## Command-line interface (CLI)

You can run `pyanimats` from the command line like so:

```
python -m pyanimats
```

Note that you must run it as a Python module, _i.e._ with `python -m`.

### CLI usage information

To see information on how to use the CLI, run

```
python -m pyanimats --help
```
