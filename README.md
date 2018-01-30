CSCI-5800: Maachine Learning

Programming Assignment 1

Implemented in Python

Assumes the 38th column is present in the file and will remove it.

It will assume `Features_Variant_1.csv` is in the same directory as `pa1.py`.

To run via Docker:

    make run

and to remove the docker image when done with it:

    make clean

To run locally (only Python 3.6 and the usual libraries (numpy, matplotlib, sklearn, and pandas) are required)

    python pa1.py

You may also specify a file with the `-f` or `--file` argument, e.g.:

    python pa1.py -f DataFile.csv

When running using the make command in docker, the file must be at the assumed location.