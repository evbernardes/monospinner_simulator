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

circle_1 = circle()
circle_2 = circle(1/np.sqrt(2))

def diff(v):
    v_ = np.diff(v)
    return np.append(v_[0], v_)

#%%
param_time['tmax'] = 12.5
param_init['init_orientation_zyz_deg'] = [90, 90, -90]
params = [param_phys, param_time, param_ctrl,  param_noise,  param_init, param_goal]
#


drift = np.array(range(-89,89+1,1))
sims = []
time_start = time.time()
for angle in drift:
    param_noise['drift_angle_init_deg'] = angle
    print(f'drift = {angle} degrees')
    sim = Monospinner(*params)
    sim.run(0.1)
    sims.append(sim)
done = True
elapsed = time.time() - time_start
elapsed_min = int(elapsed / 60)
elapsed_sec = int(elapsed - 60*elapsed_min)
print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')

#drift2 = np.array(range(-180,-90+1,1))
#sims2 = []
#time_start = time.time()
#for angle in drift2:
#    param_noise['drift_angle_init_deg'] = angle
#    print(f'drift = {angle} degrees')
#    sim = Monospinner(*params)
#    sim.run(1)
#    sims2.append(sim)
#done = True
#elapsed = time.time() - time_start
#elapsed_min = int(elapsed / 60)
#elapsed_sec = int(elapsed - 60*elapsed_min)
#print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')

#drift3 = np.array(range(90,180+1,1))
#sims3 = []
#time_start = time.time()
#for angle in drift3:
#    param_noise['drift_angle_init_deg'] = angle
#    print(f'drift = {angle} degrees')
#    sim = Monospinner(*params)
#    sim.run(1)
#    sims3.append(sim)
#done = True
#elapsed = time.time() - time_start
#elapsed_min = int(elapsed / 60)
#elapsed_sec = int(elapsed - 60*elapsed_min)
#print(f'Total elapsed time: {elapsed_min}:{elapsed_sec}')

    

#%%
width = 11.5
height = 5

#sims_ = sims[0:-1:1]
#sims_ = sims2[0:-1:1]
sims_ = (sims2 + sims + sims3)[::1]
#sims_ = [sims[0]]
#sims_ = sims
    
leg = []
aa = []
bb = []
cc = []
dd = []
init = []
plt.figure('spin drift analysis', figsize=(height*2, height*3))
plt.clf()
l = 3
c = 3
for sim in sims_:
#for sim in [sims_[0]]:
    
    leg.append(int(sim.drift_angle[0]*TODEG))
    N = sim.N
    
    r = np.linalg.norm(sim.k[:,:2], axis=1)
    r2 = r*r
#    r = np.dot(sim.k, ez)#[::-1]
    pre = np.unwrap(sim.pre)
    nut = sim.nut

    r2d = diff(r2)/sim.DT
    rd = diff(r)/sim.DT
    rdd = diff(rd)/sim.DT
    pred = diff(pre)/sim.DT
    predd = diff(pred)/sim.DT
    nutd = diff(nut)/sim.DT
    env = abs(sc.signal.hilbert(predd))
    
#    init_ = np.logical_and(sim.t > 0.1, abs(predd) < 0.1)
#    init_ = np.where(init_)[0][0]
#    init.append(init_)

    rlog = np.log(r)
#    idx = rlog > -20
    
#    rlim = 0.1
#    idx = np.logical_and(r > rlim, r < 1)
    
#    rloglim = 
    idx = np.logical_and(rlog > -20, rlog < -0.15)
#    idx = np.logical_and(idx, env < 2)
#    idx = np.logical_and(idx, abs(predd) < 10)
    idx = np.where(idx)[0]
#    rd2 = abs(rd)**2

    # find 
    deg = 1
    P = Polynomial.fit(sim.t[idx], rlog[idx], deg).convert()
    aa.append(P.coef[0])
    bb.append(P.coef[-1])
    
    drlog = diff(rlog)/sim.DT
    subplot(l,c,1,1)
    plt.plot(sim.t[idx], drlog[idx])
    plt.title(r'$\frac{d \log(r)}{dt}$')
#    plt.ylim([ min(drlog[idx]), max(drlog[idx]) ])
    
    subplot(l,c,1,2)
    plt.plot(sim.t[idx], pred[idx])
    plt.title(r'$\frac{d \phi}{dt}$')
    
    subplot(l,c,1,3)
    
#    env = abs(sc.signal.hilbert(predd[idx]))
    plt.plot(sim.t[idx], env[idx])
    plt.title(r'$\left|\frac{d^2 \phi}{dt^2} \right|$')
    
#    
    subplot(l,c,2,1)
    plt.plot(sim.t, np.log(r))
    plt.plot(sim.t[idx], P(sim.t[idx]), 'k', linestyle = 'dashed')
    plt.xlabel(r't'); plt.ylabel('log(r)')
    plt.title(r'model: $\log(r) = a + bt$')
    
#    cc.append(np.median(pred[]))
#    P2 = Polynomial.fit(sim.t[idx], pre[idx], 1).convert()
    P2 = Polynomial.fit(sim.t[:], pre[:], 1).convert()
    cc.append(P2.coef[0])
    dd.append(P2.coef[1])
    
    subplot(l,c,2,2)
    plt.plot(sim.t[:], pre[:])
    plt.plot(sim.t[:], P2(sim.t[:]), 'k', linestyle='dashed')
    plt.title(r'model: $\phi = c + dt$')
#    plt.ylim([-300, 300])
    plt.xlim([0, 13])
    
    subplot(l,c,2,3)
    plt.plot(sim.t[:], P2(sim.t[:]) - pre[:])
#    plt.plot(sim.t[:], P2(sim.t[:]), 'k', linestyle='dashed')
    plt.title(r'error of model: $\phi = c + dt$')

leg = np.array(leg)    
aa = np.array(aa)
bb = np.array(bb)
cc = np.array(cc)
dd = np.array(dd)
    

subplot(l,c,3,1)
plt.plot(leg, bb)
Pb = Polynomial.fit(leg, bb, 6)
plt.plot(leg, Pb(leg), 'k', linestyle = 'dashed')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$b$')
plt.title(r'$b = \frac{d \log(r)}{dt}$')
plt.tight_layout()

#plt.clf()
subplot(l,c,3,2)
case1 = leg < -95
case1 = np.logical_and(case1, leg )
case2 = abs(leg) < 90
case3 = leg > 95


plt.plot(leg, dd)

Pd1 = Polynomial.fit(leg[case1], dd[case1], 3)
Pd2 = Polynomial.fit(leg[case2], dd[case2], 3)
Pd3 = Polynomial.fit(leg[case3], dd[case3], 3)
plt.plot(leg[case1], Pd1(leg[case1]), 'k', linestyle = 'dashed')
plt.plot(leg[case2], Pd2(leg[case2]), 'k', linestyle = 'dashed')
plt.plot(leg[case3], Pd3(leg[case3]), 'k', linestyle = 'dashed')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$d$')
plt.title(r'$d = \frac{d \phi}{dt}$')
#plt.xlim([-90, 90])
#plt.ylim([-70, 70])
plt.tight_layout()

#plt.clf()
lamb = bb / dd
#lamb = dd / bb
deg = 1
legg = np.sign(leg)*(abs(leg)**deg)
#legg = leg
lambratio = np.mean(lamb * legg)
lambratio = -42

#P3 = Polynomial.fit(leg, lamb, 4)

subplot(l,c,3,3)
#plt.plot(leg, lambratio / np.array(leg), 'k', linestyle = 'dashed')
#plt.plot(leg, P3(leg), 'k', linestyle = 'dashed')
#plt.plot(leg, Pd(leg)/Pb(leg), 'k', linestyle = 'dashed')
plt.plot(leg, bb / dd)
plt.plot(leg, lambratio / legg, 'k', linestyle = 'dashed')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$d$')
plt.title(r'$\lambda = b / d$')
#plt.xlim([-90, 90])
#plt.ylim([-50, 50])
plt.tight_layout()


#%%#############################
plt.figure('spin drift k vector', figsize=(height*2, height*2))
plt.clf()
#ax = plt.axes(projection = '3d')
plt.axes().set_aspect('equal')

i_ = [10]
#sims_ = sims[i_]
#lambratio = -22

#for sim in sims_:
for i in i_:
    sim = sims[i]
    init = 1
    
    pre = np.unwrap(sim.pre[init:])
    r0 = np.linalg.norm(sim.k[init,:2])
    drift = sim.drift_angle_init * TODEG
#    lamb_ = lambratio / drift
    lamb_ = lambratio / drift
    r = r0 * np.exp(lamb[i] * (pre - pre[0]))
    x = r*np.cos(pre)
    y = r*np.sin(pre)
    z = np.sqrt(1 - r*r)
    
#    ax.plot3D(sim.k[:,0], sim.k[:,1], sim.k[:,2])
#    ax.plot3D(x, y, z, 'k', linestyle= 'dashed')
    
    
    plt.plot(sim.k[init:,0], sim.k[init:,1])
    plt.plot(x, y, 'k', linestyle='dashed')
#    leg.append(int(sim.drift_angle[0]*TODEG))

#plt.legend(leg)
#plt.plot(circle_2[0], circle_2[1], 'gray', linestyle='dashed')
lim = [-1.1, 1.1]
plt.xlim(lim)
plt.ylim(lim)
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
#plt.zlabel(r'$k_z$')
plt.title(r'$(k_x, k_y)$')



#plt.subplot(222)
#plt.plot(circle_2[0], circle_2[1], 'gray', linestyle='dashed')
#plt.xlim(lim)
#plt.ylim(lim)



#%%

nut_end = np.array([sim.nut[-1] for sim in sims])*TODEG
t_end = np.array([sim.t[-1] for sim in sims])

plt.figure('spin drift problem', figsize=(width, height))
plt.clf()

plt.hlines(90,0,360,'k')

for sim in sims:
    case_zero = nut_end < 5
    case_err = nut_end > 115
    case_stay = ~np.logical_or(case_zero,case_err)
    
    plt.subplot(211)
    plt.plot(drift[case_zero], nut_end[case_zero],'b.')
    plt.plot(drift[case_err], nut_end[case_err],'r.')
    plt.plot(drift[case_stay], nut_end[case_stay],'g.')

    plt.subplot(212)
#plt.hlines(90,-180,180,'k')
    plt.plot(drift[case_zero], t_end[case_zero],'b.')
    plt.plot(drift[case_err], t_end[case_err],'r.')
    plt.plot(drift[case_stay], t_end[case_stay],'g.')

#%%
#plt.figure('nutation', figsize=(11.5, 5))
#plt.clf()
##leg = []
#l = 6
#for sim in sims:
#    
#    nut = sim.nut
#    pre = sim.pre
#    nutd = diff(nut)/sim.DT
#    cnut = np.cos(sim.nut)
#    cnutd = diff(cnut)/sim.DT
#    
#    
#    subplot(l,2,1,1)
#    plt.plot(sim.t, nut*TODEG)
#    plt.title(r'$\theta$')
#    
#    subplot(l,2,2,1)
#    plt.plot(sim.t, nutd)
#    plt.title(r'$\frac{d\theta}{dt}$')
#    
#    subplot(l,2,1,2)
#    plt.plot(sim.t, cnut)
#    plt.title(r'$\cos(\theta)$')
#    
#    subplot(l,2,2,2)
#    plt.plot(sim.t, cnutd)
#    plt.title(r'$\frac{d\cos(\theta)}{dt}$')
#    
#    
#    subplot(l,2,3,1)
#    plt.plot(sim.t, np.log(nut))
#    plt.title(r'$\log(\theta)$')
#    
#    subplot(l,2,3,2)
#    plt.plot(sim.t, np.log(cnut))
#    plt.title(r'$\log \left(\cos(\theta)\right)$')
#    plt.ylim([-2, 0.1])
#    
#    subplot(l,2,4,1)
#    plt.plot(sim.t, np.log(-nutd))
#    plt.title(r'$\log \left( -\frac{d\theta}{dt} \right)$')
#    
#    subplot(l,2,4,2)
#    plt.plot(sim.t, np.log(cnutd))
#    plt.title(r'$\log \left( \frac{d\cos(\theta)}{dt} \right)$')
#    
#    subplot(l,2,5,1)
#    plt.plot(sim.t, diff(np.log(nut))/sim.DT)
#    plt.title(r'$\frac{d \log(\theta) }{dt}$')
#    
#    subplot(l,2,5,2)
#    plt.plot(sim.t, diff(np.log(cnut))/sim.DT)
#    plt.title(r'$\frac{d \log(\cos(\theta)) }{dt}$')
#    plt.ylim([-0.1, 1])
#    
#    subplot(l,2,6,1)
##    plt.plot(pre*TODEG, np.log(nut))
##    plt.title(r'$\log(\theta) / \phi$')
#    
#subplot(l,2,1,2)
##plt.legend(drift)
#plt.tight_layout()
#
#plt.figure('curvature', figsize=(width, height))
#plt.clf()
#for sim in sims:
#    plt.subplot(211)
#    c = curvature(sim.k, sim.DT)
#    plt.plot(sim.t, c)
#    plt.subplot(212)
#    
#    pre = np.unwrap(sim.pre)
#    pred = np.append(0, np.diff(pre))/sim.DT
#    plt.plot(sim.t, pred)
#plt.legend(drift)




