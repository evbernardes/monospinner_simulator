#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:20:49 2021

@author: evbernardes
"""
#%%
import time
import scipy as sc
import numpy as np
from numpy.linalg import norm
from numpy.polynomial import Polynomial
#from scipy import integrate
import matplotlib.pyplot as plt
#import quaternionic as quat
from helpers import TODEG, TORAD, ez #, Id, zeroM, zeroV, ez, Ez, eps
from helpers import wrap, circle, curvature
from helpers_plot import plot_x_eta, plot_k_omega, plot_F, subplot
from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_system import Monospinner
from mpl_toolkits.mplot3d import Axes3D
#import os

#%%
param_time['tmax'] = 12.5
param_init['init_orientation_zyz_deg'] = [90, 90, -90]

tmax = param_time['tmax']
zyz_deg = param_init['init_orientation_zyz_deg']
dirs = f'nonoise_{tmax}s_{zyz_deg[1]}deg'

params = [param_phys, param_time, param_ctrl,  param_noise,  param_init, param_goal]
#

#drift = np.array(range(-89,89+1,1))
#sims = []
#time_start = time.time()
#for angle in drift:
#    param_noise['drift_angle_init_deg'] = int(angle)
#    print(f'drift = {angle} degrees')
#    sim = Monospinner(*params)
#    sim.run(0.1)
#    sims.append(sim)
#    sim.save(f'saved_data/{dirs}/{angle}.json')
##    sim.save('test.json')
#done = True
#elapsed = time.time() - time_start
#elapsed_min = int(elapsed / 60)
#elapsed_sec = int(elapsed - 60*elapsed_min)
#print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')

#drift2 = np.array(range(-180,-90+1,1))
#sims2 = []
#time_start = time.time()
#for angle in drift2:
#    param_noise['drift_angle_init_deg'] = int(angle)
#    print(f'drift = {angle} degrees')
#    sim = Monospinner(*params)
#    sim.run(1)
#    sims2.append(sim)
#    sim.save(f'saved_data/{dirs}/{angle}.json')
#    
#done = True
#elapsed = time.time() - time_start
#elapsed_min = int(elapsed / 60)
#elapsed_sec = int(elapsed - 60*elapsed_min)
#print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')

drift3 = np.array(range(90,180+1,1))
sims3 = []
time_start = time.time()
for angle in drift3:
    param_noise['drift_angle_init_deg'] = int(angle)
    print(f'drift = {angle} degrees')
    sim = Monospinner(*params)
    sim.run(1)
    sims3.append(sim)
    sim.save(f'saved_data/{dirs}/{angle}.json')
done = True
elapsed = time.time() - time_start
elapsed_min = int(elapsed / 60)
elapsed_sec = int(elapsed - 60*elapsed_min)
print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')
