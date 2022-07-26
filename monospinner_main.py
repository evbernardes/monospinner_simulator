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
#from scipy import integrate
import matplotlib.pyplot as plt
#import quaternionic as quat
from helpers import TODEG, TORAD, Id, zeroM, zeroV, ez, Ez, eps
from helpers import wrap, circle, diff, angdiff
from helpers_plot import plot_x_eta, plot_k_omega, plot_F, subplot
from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_system import Monospinner
#import os

def get_table(data_dict):
    for key in data_dict.keys():
        print(f'${key}$ & ${data_dict[key]:.10f}$ \\\\') 
        print('\\hline')

def plot_circle(r = 1, *args):
    circ = circle(r)
    plt.plot(circ[0], circ[1], *args)
    
circle_1 = circle()
circle_2 = circle(1/np.sqrt(2))

def dot(a, b):
    return np.sum(a*b, axis=1)
#%%
#
drift = 0
drift_rate = 0.2

#param_goal = {
#        'alpha_deg': 20,
#        'beta_deg': 0}
param_phys['angvelxy_coef'] = 0
## 
##
#param_goal = {
#        'alpha_deg': [0]+[45]*8+[0, 0],
#        'beta_deg': [0, 90, 90, 180, 180, 270, 270, 0, 0, 0, 0]}
#param_phys['hs'] = 0
param_phys['hs'] *= 1
param_init = {
        'spin_percentage': 0.0,
        'init_orientation_zyz_deg': [0, 90, 0],
        'init_position': [0, 0, 0],
        }

param_noise['drift_angle_init_deg'] = drift
param_noise['drift_rate_rads'] = drift_rate
#param_noise['drift_flip'] = True
#param_noise['noise_measures'] = [0.0001, 0.0001]
param_noise['noise_random'] = [0.00, 0.00]
param_noise['noise_impulses'] = [15, 15]
param_noise['number_impulses'] = 00

param_time['tmax'] = 50
param_ctrl['Kp'] = 0.7
param_ctrl['alpha_lim_deg'] = [0, 15]
param_ctrl['window'] = 0
sim = Monospinner(param_phys, param_time, param_ctrl,  param_noise,  param_init, param_goal)
sim.run()

width = 5
height = 5
i = sim.i

filename = 'test.json'
#sim.save(filename)
#sim = Monospinner.load(filename)
#%%

kk = np.cross(sim.k, sim.kd)
K = np.array([q_.inverse.rotate(k_) for q_, k_ in zip(sim.q, kk)])
#%% drift calc
k = sim.k#_measured
kd = sim.kd
kD = k - kd
#r = norm((k-kd)[:,:2], axis=1)
#pre_ = sim.pre
#pre_ = np.arctan2((k-kd).T[1],(k-kd).T[0])
#r2 = r*r
#b = -diff(np.log(r), sim.DT)
#d = diff(pre_, sim.DT)
#case = b < 0
#

kdot = np.diff(k, axis=0) / sim.DT
kdot = np.vstack([kdot[0], kdot])

#b = kD.T[2] * kdot.T[2] - dot(kD, kdot)
#d = dot(ez, np.cross(kD, kdot))

#b = kD.T[2] * kdot.T[2] - dot(kD, kdot)
#b = dot(kD,ez)*dot(kdot,ez) - dot(kD, kdot)
#b = dot(np.cross(kdot,ez), np.cross(kD,ez))
b = -dot(np.cross(kdot,ez), np.cross(kD,ez))
d = dot(ez, np.cross(kD, kdot))
#d = k.T[0] * kdot.T[1] - k.T[1] * kdot.T[0]

#b = 

z = d + sc.signal.hilbert(d)
#d = np.sign(d) * abs(z)


#d = np.sqrt(d*d + d_*d_)

#valid = (d*d + b*b) > (10e-14)**2
valid = norm(kD,axis=1) > 1e-7
#valid = norm(kD,axis=1) > 0.9
#valid = np.logical_and(valid, norm(kD,axis=1) < 1)
#valid = (d*d + b*b) > (0)**2

#valid = np.logical_and(valid, np.abs(d) < 50)


drift_estimation = np.arctan2(d, b)



fb, fa = sc.signal.butter(5, 0.02)
b_filt = sc.signal.lfilter(fb, fa, b)
d_filt = sc.signal.lfilter(fb, fa, d)
drift_estimation_f1 = wrap(sc.signal.lfilter(fb, fa, np.unwrap(drift_estimation)))
drift_estimation_f2 = np.arctan2(d_filt, b_filt)


err = angdiff(drift_estimation, sim.drift_angle)
err_filt = angdiff(drift_estimation_f1, sim.drift_angle)

r_ = np.sqrt(b*b + d*d)
r_filt = np.sqrt(b_filt*b_filt + d_filt*d_filt)

t = sim.t
plt.figure('drift estimation')
plt.clf()
subplot(5111)
plt.plot(t, wrap(sim.drift_angle)*TODEG, 'k')
#plt.plot(t, drift_estimation * TODEG, 'b.')
plt.plot(t[valid], drift_estimation[valid] * TODEG, 'r.')
xlim = plt.xlim()
plt.hlines(90, xlim[0], xlim[1], colors='gray', linestyle='dashed')
plt.hlines(-90, xlim[0], xlim[1], colors='gray', linestyle='dashed')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta \theta$ (spin drift)')
plt.title('drift rate: 1 rad/s')

subplot(5112)
plt.plot(sim.t, abs(err), 'r.')
plt.plot(sim.t, abs(err_filt), 'b.')
plt.title(r'$|err(t)|$')
plt.legend(['no filter','filter'])


subplot(5113)
plt.plot(sim.t, r_, 'r.')
plt.plot(sim.t, r_filt, 'b.')
plt.title(r'$\sqrt{a^2+b^2}(t)$')
plt.legend(['no filter','filter'])

subplot(5114)
plt.plot(sim.t, b, 'k')
#plt.plot(sim.t, b/r2, 'b')
#plt.plot(sim.t, b_filt, 'b')
plt.legend(['b'])
subplot(5115)
plt.plot(sim.t, d, 'k')
#plt.plot(sim.t, d/r2, 'b')
#plt.plot(sim.t, d_filt, 'b')
plt.legend(['d'])


plt.tight_layout()


plt.figure('drift estimation')
plt.clf()
subplot(5111)
plt.plot(t, wrap(sim.drift_angle)*TODEG, 'k')
#plt.plot(t, drift_estimation * TODEG, 'b.')
plt.plot(t[valid], drift_estimation[valid] * TODEG, 'r.')
xlim = plt.xlim()
plt.hlines(90, xlim[0], xlim[1], colors='gray', linestyle='dashed')
plt.hlines(-90, xlim[0], xlim[1], colors='gray', linestyle='dashed')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta \theta$ (spin drift)')
plt.title('drift rate: 1 rad/s')


#plt.figure('b and d')
#plt.clf()
#subplot(2211)
##plt.plot(r_, err, 'r.')
##plt.plot(r_, 'r.')
##plt.plot((b/r_)[case], (d/r_)[case], 'b.')
##plt.plot((b/r_)[~case], (d/r_)[~case], 'r.')
##plt.xlim([-1.1, 1.1])
##plt.ylim([-1.1, 1.1])
#subplot(2212)
#plt.plot((b)[case], (d)[case], 'b.')
#plt.plot((b)[~case], (d)[~case], 'r.')
#plt.xlim([-5, 5])
#plt.ylim([-20, 5])
#subplot(2221)
#plt.plot(sim.t, b, 'k')
#plt.plot(sim.t, b_filt)
#plt.legend(['b'])
#subplot(2222)
#plt.plot(sim.t, d, 'k')
#plt.plot(sim.t, d_filt)
#plt.legend(['d'])

#%%
#plt.figure('spin drift k vector', figsize=(5, 5))
plt.figure('k vector top', figsize=(5, 5))
plt.clf()
plt.axes().set_aspect('equal')
#subplot(1211)
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

plt.figure('k vector', figsize=(5,5))
plt.clf()
subplot(3,1,1,1); 
plt.plot(t, sim.kd[:, 0], 'k'); plt.ylabel(r'$k_x$');
plt.plot(t, sim.k[:, 0],'r'); plt.ylim([-0.5, 0.5]);
plt.title(r'middle vector $(k)$')
#    plt.hlines(kd[0], t[0], t[-1], colors='k', linestyles='dashed')
subplot(3,1,2,1); 
plt.plot(t, sim.kd[:, 1], 'k'); plt.ylabel(r'$k_y$');
plt.plot(t, sim.k[:, 1],'g'); plt.ylim([-0.5, 0.5])
#    plt.hlines(kd[1], t[0], t[-1], colors='k', linestyles='dashed')
subplot(3,1,3,1); 
plt.plot(t, sim.kd[:, 2], 'k'); plt.ylabel(r'$k_z$');
plt.plot(t, sim.k[:, 2],'b'); plt.ylim([0.8, 1.1])
plt.tight_layout()

#plt.subplot(222)
#plt.plot(np.unwrap(sim.pre), sim.nut)

#%%

q = sim.q
k = sim.k
w = sim.w
Rw = np.array([q_.rotate(w_) for q_,w_ in zip(q,w)])

pre,nut,spin = q.to_euler_angles.T
spindot2 = diff(pre + spin)/sim.DT
spindot2 = np.hstack([spindot2[0], spindot2])

spindot = dot(k, Rw) / k.dot(ez)

wb = (ez[np.newaxis].T * spindot).T

wk = np.array([q_.rotate(w_) for q_,w_ in zip(q, w - wb)])

#%%
#ctrlorder = [1,2,5,10]
#psi=np.linspace(0, np.pi*0.95, 100)
#T = np.tan(psi/2)
#magnitude = [T / (np.cos(psi/2)**(k-1)) for k in ctrlorder]
#plt.figure('ctrlorder')
#plt.clf()
#for m in magnitude:
#    plt.plot(psi*180/np.pi, m)
#plt.legend(ctrlorder)
#plt.ylim(0, magnitude[0][-1])
#plt.ylabel(r'$m$')
#plt.xlabel(r'$\psi$ (deg)')

#%%
t = sim.t
w = sim.w
k = sim.k
kd = sim.kd

plt.figure('sim results', figsize=(15, 5))
plot_k_omega(sim)
subplot(3312)
plt.ylim([-0.5, 0.5])
subplot(3322)
plt.ylim([-0.5, 0.5])
subplot(3332)
plt.ylim([0.8, 1.1])

plt.figure('angular velocity', figsize=(5, 5))
plt.clf()
subplot(3111)
plt.plot(t, w.T[0], 'r'); plt.ylabel(r'$\omega_x$')
subplot(3121)
plt.plot(t, w.T[1], 'g'); plt.ylabel(r'$\omega_y$')
subplot(3131)
#plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
plt.plot([t[0], t[-1]], [sim.terminal_wz, sim.terminal_wz], 'k', linestyle='dashed')
plt.plot(t, w.T[2], 'b'); plt.xlabel(r'$t$ (seconds)'); plt.ylabel(r'$\omega_z$')
plt.legend(['terminal velocity'])
plt.tight_layout()

plt.figure('motor inputs', figsize=(5, 5*2/3))
plt.clf()
#subplot(2111)
#plt.plot(t, sim.angles.T[0]*TODEG, 'r'); plt.ylabel(r'$\beta$ (deg)')
subplot(2111)
plt.plot(t, sim.angles.T[1]*TODEG, 'g'); plt.ylabel(r'$\alpha$ (deg)')
subplot(2121)
#plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
#plt.plot([t[0], t[-1]], [sim.terminal_wz, sim.terminal_wz], 'k', linestyle='dashed')
plt.plot(t, sim.rotvel, 'b'); plt.xlabel(r'$t$ (seconds)'); plt.ylabel(r'$\dot \gamma$ (rad/$s$)')
plt.tight_layout()

plt.figure('beta', figsize=(10, 2))
plt.clf()
#subplot(2111)
plt.plot(t, sim.angles.T[0]*TODEG, 'r'); plt.ylabel(r'$\beta$ (deg)')
#subplot(2121)
#plt.plot(t, sim.spin*TODEG, 'r'); plt.ylabel(r'$\beta$ (deg)')
plt.tight_layout()

#plt.figure('motor inputs', figsize=(5, 5))
#plt.clf()
##subplot(3111)
##plt.plot(t, sim.angles.T[0]*TODEG, 'r'); plt.ylabel(r'$\beta$ (deg)')
#subplot(3111)
#plt.plot(t, sim.angles.T[1]*TODEG, 'g'); plt.ylabel(r'$\alpha$ (deg)')
#subplot(3121)
##plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
##plt.plot([t[0], t[-1]], [sim.terminal_wz, sim.terminal_wz], 'k', linestyle='dashed')
#plt.plot(t, sim.rotvel, 'b'); plt.ylabel(r'$\dot \gamma$ (rad/$s$)')
#subplot(3122)
##plt.hlines(sim.terminal_wz, t[0], t[-1], 'k', linestyles='dashed')
##plt.plot([t[0], t[-1]], [sim.terminal_wz, sim.terminal_wz], 'k', linestyle='dashed')
#a = np.logical_and(t > 0.8, t < 1.05)
#plt.plot(t[a], sim.rotvel[a], 'b'); plt.ylabel(r'$\dot \gamma$ (rad/$s$)')
#
#plt.xlabel(r'$t$ (seconds)'); 
#plt.tight_layout()