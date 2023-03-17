# Monospinner simulator
Python implementation of a simulation of the drone described in the paper:

- **Evandro Bernardes, Fréféric Boyer, and Stéphane Viollet. Modelling, control and simulation of a single rotor uav with swashplateless
torque modulation** (Elsevier Aerospace Science and Technology, in revision)

## Tested on Linux with:
- `Python 3.8.10`
- `numpy 1.21.6`
- `quaternionic 1.0.5`
- `blender 3.4.1`

## How to run simulations

### To use a new parameters descriptor file:
- Make a copy of `parameters/default.yaml` to something different, like `parameters/sim_1.yaml` for example.
- Change newly created parameters file with desired parameters.
- Open `main.py`
- Load newly created `parameters` file into main and run

### Alternatively, default parameters can be used
- Open `main.py`
- Load `parameters/default.yaml`
- Changed only the designed parameters and run

## Saving and loading
The `Monospinner` class implements `save` and `load` methods to store the whole simulation in `JSON` format.

This can be used, for example, if new plots are to be made out of a previously run simulation.

## Visualization
A hacky solution to see the visualization is implemented with the `play` method, but this is not very stable.

## TODO:
This code must be completely refactored.
It was implemented in a quick-and-dirty fashion, with some sub-optimal implementations that could be replace for better tested functions.
