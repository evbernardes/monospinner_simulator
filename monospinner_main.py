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
import matplotlib.pyplot as plt
from helpers import TODEG, TORAD, Id, zeroM, zeroV, ez, Ez, eps
from helpers import wrap, circle, diff, angdiff
from helpers_plot import plot_x_eta, plot_k_omega, plot_F, subplot
from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_system import Monospinner

#%% Change some parameters and run simulation
param_goal = {
        'alpha_deg': [0]+[45]*8+[0, 0],
        'beta_deg': [0, 90, 90, 180, 180, 270, 270, 0, 0, 0, 0]}

param_noise['noise_random'] = [0.005, 0.005]
param_noise['noise_impulses'] = [1, 1]
param_noise['number_impulses'] = 9

param_time['tmax'] = 15
param_ctrl['Kp'] = 0.5
param_ctrl['alpha_lim_deg'] = [0, 10]
param_ctrl['window'] = 0

sim = Monospinner(param_phys, param_time, param_ctrl,  param_noise,  param_init, param_goal)
sim.run()

#%% PLOTS
t = sim.t
width = 5
height = 5
i = sim.i

### sim results
circle_1 = circle()
circle_2 = circle(1/np.sqrt(2))
plt.figure('k vector top', figsize=(5, 5))
plt.clf()
plt.axes().set_aspect('equal')
legend = []
plt.plot(circle(1)[0], circle(1)[1], 'k', linestyle='dashed'); legend.append(r'$\theta_2 = \pi$')
plt.plot(circle(np.cos(np.pi/4))[0], circle(np.cos(np.pi/4))[1], 'gray', linestyle='dashed'); legend.append(r'$\theta_2 = \pi/2$')
plt.plot(sim.k[:,0], sim.k[:,1]); legend.append('middle vector')
plt.plot(sim.kd[0].T[0], sim.kd[0].T[1], 'r.'); legend.append('initial orientation')
plt.legend(legend)
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.title(r'$(k_x, k_y)$')
plt.tight_layout()

### sim results
plt.figure('sim results', figsize=(15, 5))
plot_k_omega(sim)
subplot(3312)
plt.ylim([-0.5, 0.5])
subplot(3322)
plt.ylim([-0.5, 0.5])
subplot(3332)
plt.ylim([0.8, 1.1])

### sim results in ZYZ angles
plt.figure('zyz', figsize=(5,5))
plt.clf()

pred = np.arctan2(sim.nd.T[1],sim.nd.T[0])
nutd = np.arccos(sim.nd.T[2])

pre = np.copy(sim.pre)
pre[abs(sim.nut) < 0.001] = 0

subplot(3,1,1,1);
plt.plot(t, pred*TODEG, 'k')
plt.plot(t, wrap(pre)*TODEG,'r');
plt.ylim([-200, 200]);
subplot(3,1,2,1);
plt.plot(t, nutd*TODEG, 'k')
plt.plot(t, wrap(sim.nut)*TODEG,'g');
plt.ylim([-10, 90]);
subplot(3,1,3,1);
plt.plot(t, wrap(sim.spin)*TODEG,'b');
plt.tight_layout()
