#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:20:49 2021

@author: evbernardes
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt

from Simulator.monospinner import Monospinner
from Simulator import helpers_plot

# %% open parameters
parameters_filenames = [
        'default.yaml',
        'sim1.yaml',
        'sim2.yaml']

with open('parameters/'+parameters_filenames[1]) as f:
    parameters = yaml.load(f, Loader=yaml.SafeLoader)

# To change simulation, either load a different file or change just some of
# the parameters, for example:
# parameters['noise_random'] = [0.005, 0.005]

# When changing the desired orientation angles, remember that they are given
# in degrees and that len(alpha_deg) == len(beta_deg)

# to save the parameters, use:
# with open('parameters/custom.yaml') as f:
#     yaml.dump(parameters, sort_keys=True)

#sim = Monospinner(parameters)
#sim.run()
sim = Monospinner.load('article_data/sim2.json')

# after running, simulations can be saved as json files:
# sim.save('saved_data/simulation_data.json')
# and then, they can be reloaded
# sim = Monospinner.load('saved_data/simulation_data.json')

# %% plots of sim results
plt.figure('nmiddle projection', figsize=(5, 5))
helpers_plot.plot_nmiddle_projection(sim)

# sim results
plt.figure('system overview', figsize=(15, 5))
helpers_plot.plot_nmiddle_and_angvel(sim)

# sim results in ZYZ angles
plt.figure('zyz', figsize=(5, 5))
helpers_plot.plot_precession_nutation(sim)
