# PF Hub Benchmark 3 code using Firedrake library
This is the code for Dendritic Growth benchmark.
The domain is a square with no-flux bcs.

Follow the link for the description of the benchmark:
https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/

## Installation
First install [Firedrake](https://firedrakeproject.org/download.html).
Then activate firedrake virtual environment with:

    source firedrake/bin/activate

After that install the firedrake-ts with:

    python -m pip install git+https://github.com/IvanYashchuk/firedrake-ts.git@master

## Running the code

Run the code with:

    mpirun -n 4 python benchmark3.py -firedrake_ts_monitor -firedrake_snes_monitor -firedrake_ts_adapt_type basic -firedrake_ts_adapt_monitor

Above command will run the simulation with progress output using basic timestepping adaptation from PETSc TS library.
The default timestepping algorithm is backward Euler method.
Other timesteppers can be used by passing `-firedrake_ts_type TSTYPE`. All implicit methods of PETSc TS are supported.
