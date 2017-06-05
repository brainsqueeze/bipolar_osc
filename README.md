bipolar_osc
===========

Code for computing bipolar neutrino oscillations from analytic ansatz in core-collapse supernovae. This is based on the work in [https://inspirehep.net/record/1419546?ln=en](https://inspirehep.net/record/1419546?ln=en).

Before executing, do
```sh
mkdir data
```
in the same directory as the python code.  To execute just run
```sh
python src/integrate.py
```
in the terminal.

Output files for the function Log(10, Lambda) at each distance are stored in the
```sh
data
```
directory.
