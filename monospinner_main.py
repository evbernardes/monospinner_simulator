#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:20:49 2021

@author: evbernardes
"""
#%%
import time
import numpy as np
from numpy.linalg import norm
#from scipy import integrate
import matplotlib.pyplot as plt
#import quaternionic as quat
from helpers import TODEG, TORAD, Id, zeroM, zeroV, ez, Ez, eps
from helpers import wrap, circle
from helpers_plot import plot_x_eta, plot_k_omega, plot_F
from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_system import Monospinner
#import os

circle_1 = circle()
circle_2 = circle(1/np.sqrt(2))

#%%
drift = 50
param_noise['drift_angle_init_deg'] = drift
sim = Monospinner(param_phys, param_time, param_ctrl,  param_noise,  param_init, param_goal)
sim.run(1)

width = 11.5
height = 5
i = sim.i

filename = 'test.json'
sim.save(filename)
sim = Monospinner.load(filename)

#%%
plt.figure('body pos and vel', figsize=(width, height)); plot_x_eta(sim)
plt.figure('k and omega', figsize=(width, height/1.2)); plot_k_omega(sim)

#%%
i = sim.i
def get_lims(forces, name_ignore):
    min_ = np.zeros(3)
    max_ = np.zeros(3)
    for force_, name in forces:
        if name not in name_ignore:
            min_ = np.vstack([min_, np.min(force_,axis=0)])
            max_ = np.vstack([max_, np.max(force_,axis=0)])
    return list(np.vstack([np.min(min_,axis=0), np.max(max_,axis=0)]).T)

torques = [[sim.t_imp,r'imp'],
           [sim.t_rndn,r'rdnd'],
           [sim.t_ctrl,r'ctrl'],
           [sim.t_grav,r'grav'],
           [sim.t_prop,r'prop'],
           [sim.t_aero,r'aero']]

torque_lim = get_lims(torques, [r'ctrl', r'impa'])

forces =  [[sim.f_imp,r'imp'],
           [sim.f_rndn,r'rndn'],
           [sim.f_ctrl,r'ctrl'],
           [sim.f_grav,r'grav'],
           [sim.f_prop,r'prop']]

force_lim = get_lims(forces, [r'ctrl', r'imp'])

plt.figure('torques and forces', figsize=(width, height))
#plot_F(t[:i-1], torques, forces, torque_lim = torque_lim, force_lim = force_lim)
plot_F(sim.t, torques, forces, torque_lim = None, force_lim = None)


#%%
plt.figure('spin drift k vector', figsize=(height*2, height*2))
plt.clf()
plt.axes().set_aspect('equal')
plt.subplot(221)
plt.plot(sim.k.T[0], sim.k.T[1])
plt.legend([drift])
#plt.clf()
plt.plot(circle_1[0], circle_1[1], 'k', linestyle='dashed')
plt.plot(circle_2[0], circle_2[1], 'gray', linestyle='dashed')
#plt.plot(np.array(circle()), 'k', linestyle='dashed')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.title(r'$(k_x, k_y)$')

plt.subplot(222)
plt.plot(np.unwrap(sim.pre), sim.nut)





