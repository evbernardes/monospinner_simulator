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
from helpers import wrap, circle, curvature, diff, fit_sin
from helpers_plot import plot_x_eta, plot_k_omega, plot_F, subplot
from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_system import Monospinner
from mpl_toolkits.mplot3d import Axes3D
import os

circle_1 = circle()
circle_2 = circle(1/np.sqrt(2))

def bars(x = [-90,0,90]):
    xlim = plt.xlim()
    ylim = plt.ylim()
    for x_ in x:
        plt.vlines(x_,ylim[0],ylim[1],colors='gray',linestyle = 'dashed')
    plt.xlim(xlim)

#%%
folder = 'saved_data/nonoise_12.5s_90deg'

files = os.listdir('saved_data/nonoise_12.5s_90deg')
drift_angle = np.array([int(file[:-5]) for file in files])
drift_angle.sort()

#%% loading
time_start = time.time()
#sims = [Monospinner.load(f'{folder}/{angle}.json') for angle in drift_angle]
sims = {angle : Monospinner.load(f'{folder}/{angle}.json') for angle in drift_angle}

elapsed = time.time() - time_start
elapsed_min = int(elapsed / 60)
elapsed_sec = int(elapsed - 60*elapsed_min)
print(f'Total loading time: {elapsed_min}:{elapsed_sec}')
#%%
width = 11.5
height = 5

#select = [-85, 85, 10]
#select = [-170, 170, 1]
#select = [0, 170, 1]
#case = np.logical_and(drift_angle >= select[0], drift_angle <= select[1])
#sims_ = np.array(sims)[case][::select[2]].tolist()

#sims_ = sims[:]

#angles_ = drift_angle[:]
angles_ = np.array([-150, -130, -110, -90, -80])
    
leg = []
aa = []
bb = []
cc = []
dd = []
init = []
plt.figure('spin drift analysis', figsize=(height*2, height*3))
plt.clf()
l = 4
c = 3

for angle in angles_:
#for sim in [sims_[0]]:
    sim = sims[angle]
    leg.append(int(sim.drift_angle[0]*TODEG))
    N = sim.N
    
    r = np.linalg.norm(sim.k[:,:2], axis=1)
#    r2 = r*r
    kz = sim.k.T[-1]
#    r = np.dot(sim.k, ez)#[::-1]
    pre = np.unwrap(sim.pre)
    nut = sim.nut
    
#    r = kz

#    r2d = diff(r2)/sim.DT
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
    
    idx = np.logical_and(rlog > -20, rlog < -0.2)
#    idx = np.logical_and(rlog > -100, rlog < 100)
    idx = np.where(idx)[0]
#    rd2 = abs(rd)**2

    # find 
    deg = 1
    P = Polynomial.fit(sim.t[idx], rlog[idx], deg).convert()
    aa.append(P.coef[0])
    bb.append(P.coef[-1])
    
    drlog = diff(rlog)/sim.DT
#    subplot(l,c,1,1)
#    plt.plot(sim.t[idx], drlog[idx])
#    plt.title(r'$\frac{d \log(r)}{dt}$')
##    plt.ylim([ min(drlog[idx]), max(drlog[idx]) ])
    
    
    subplot(l,c,1,2)
    plt.plot(sim.t[idx], drlog[idx])
    plt.title(r'$\frac{d \log(r)}{dt}$')
#    plt.ylim([ min(drlog[idx]), max(drlog[idx]) ])
    
    subplot(l,c,1,3)
    plt.plot(sim.t[idx], pred[idx])
    plt.title(r'$\frac{d \phi}{dt}$')
    
    subplot(l,c,2,1)
    plt.plot(sim.t, r)
    plt.plot(sim.t[idx], np.exp(P(sim.t[idx])), 'k', linestyle = 'dashed')
    plt.xlabel(r't'); plt.ylabel('r')
    plt.title(r'model: $r = \exp(a + bt)$')
    
    subplot(l,c,2,2)
    plt.plot(sim.t, np.log(r))
    plt.plot(sim.t[idx], P(sim.t[idx]), 'k', linestyle = 'dashed')
    plt.xlabel(r't'); plt.ylabel('log(r)')
    plt.title(r'model: $\log(r) = a + bt$')
    
#    cc.append(np.median(pred[]))
#    P2 = Polynomial.fit(sim.t[idx], pre[idx], 1).convert()
    P2 = Polynomial.fit(sim.t[:], pre[:], 1).convert()
    cc.append(P2.coef[0])
    dd.append(P2.coef[1])
    
    subplot(l,c,2,3)
    plt.plot(sim.t[:], pre[:])
    plt.plot(sim.t[:], P2(sim.t[:]), 'k', linestyle='dashed')
    plt.title(r'model: $\phi = c + dt$')
#    plt.ylim([-300, 300])
    plt.xlim([0, 13])
    
    

leg = np.array(leg)    
aa = np.array(aa)
bb = np.array(bb)
cc = np.array(cc)
dd = np.array(dd)
    

subplot(l,c,3,1)
plt.plot(leg, bb)
Pb = Polynomial.fit(leg, bb, 6)
Sfit = fit_sin(leg, bb)

plt.plot(leg, Pb(leg), 'k', linestyle = 'dashed')
plt.plot(leg, Sfit['fitfunc'](leg), 'r', linestyle = 'dashed')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$b$')
plt.title(r'$b = \frac{d \log(r)}{dt}$')
plt.tight_layout()
plt.legend(['data', f'pol deg {Pb.degree()}', 'sine fit'])
bars()

#plt.clf()
case1 = leg < -95
case1 = np.logical_and(case1, leg )
case2 = abs(leg) < 90
case3 = leg > 95
#Pd1 = Polynomial.fit(leg[case1], dd[case1], 3)
#Pd2 = Polynomial.fit(leg[case2], dd[case2], 3)
#Pd3 = Polynomial.fit(leg[case3], dd[case3], 3)
#plt.plot(leg[case1], Pd1(leg[case1]), 'k', linestyle = 'dashed')
#plt.plot(leg[case2], Pd2(leg[case2]), 'k', linestyle = 'dashed')
#plt.plot(leg[case3], Pd3(leg[case3]), 'k', linestyle = 'dashed')
#Pd = Polynomial.fit(leg[10:], dd[10:], 30)

subplot(l,c,3,2)
plt.plot(leg, dd)
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$d$')
plt.title(r'$d = \frac{d \phi}{dt}$')
bars()
plt.tight_layout()

#P3 = Polynomial.fit(leg, lamb, 4)

#subplot(l,c,3,3)
#plt.plot(leg, 1 / (diff(dd)/diff(leg)))
#plt.xlabel(r'$\Delta \theta$ (spin drift)')
#plt.ylabel(r'$\dot d$')
#bars()


subplot(l,c,4,1)
plt.plot(leg, bb / dd)
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$d$')
plt.title(r'$\lambda = b / d$')
bars()

subplot(l,c,4,2)
plt.plot(leg, dd / bb)
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.ylabel(r'$d$')
plt.title(r'$\lambda = d / b$')
bars()

subplot(l,c,3,3)
#plt.plot(leg, np.arctan2(bb, dd), 'r')

test1 = np.arctan2(-dd, bb)*TODEG
test2 = np.arctan(-dd / bb)*TODEG
testcase = np.logical_and(leg > 10, leg < 80)
Ptest1 = Polynomial.fit(leg[testcase], test1[testcase], 1)
testcase = np.logical_and(leg > -80, leg < 80)
Ptest2 = Polynomial.fit(leg[testcase], test2[testcase], 1)
#plt.plot(leg, test1, 'g.')
plt.plot(leg, test2, 'b.')
#plt.plot(leg, Ptest1(leg), 'y')
plt.plot(leg, Ptest2(leg), 'k')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
plt.title(r'$\tan^{-1}\left( \frac{d}{b} \right)$')
#plt.legend(['atan2(d,b)','atan(d/b)'])
plt.legend(['data',r'fit: $\tan^{-1}\left( \frac{d}{b} \right) = -\Delta \theta$'])
bars()

subplot(l,c,4,3)
#plt.plot(test1, leg, 'g.')
plt.plot(test2, leg, 'b.')
#plt.ylim(ylim)
#plt.plot(test2, leg, 'b.')
plt.xlabel(r'$\Delta \theta$ (spin drift)')
#plt.legend(['atan2(b,d)','atan2(d,b)','atan(d/b)'])
plt.legend(['atan2(d,b)','atan(d/b)'])
bars()



#plt.title(r'$\lambda = d / b$')

lamb = bb / dd
deg = 1
#legg = np.sign(leg)*(abs(leg)**deg)
#lambratio = np.mean(lamb * legg)
#lambratio = -42

plt.tight_layout()

#%%
r = np.sqrt(bb*bb + dd*dd)
a = np.arctan(dd/bb)
plt.figure('dd bb', figsize=(width, width))
plt.clf()
subplot(2211)
plt.plot(bb/r, dd/r)
plt.plot(circle_1[0], circle_1[1], 'gray', linestyle = 'dashed')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
subplot(2212)
plt.plot(bb, dd)
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.plot(circle(2)[0], circle(2)[1], 'gray', linestyle = 'dashed')
subplot(2221)
plt.plot(leg, r)
plt.ylabel('r')
bars()
subplot(2222)
plt.plot(leg, np.tan(leg*TORAD), 'k')
plt.plot(leg, -dd/bb, '.')
plt.ylim([-400, 400])
plt.ylabel('arctan')
bars()

#%%#############################
plt.figure('spin drift k vector', figsize=(height*2, height*2))
plt.clf()
#ax = plt.axes(projection = '3d')
plt.axes().set_aspect('equal')

#i_ = [10]
i_ = np.where(leg == 80)[0]
#sims_ = sims[i_]
#lambratio = -22

#for sim in sims_:
for i in i_:
    sim = sims_[i]
    init = int(sim.N/10)
#    init = 1
    
    pre = np.unwrap(sim.pre[init:])
    r0 = np.linalg.norm(sim.k[init,:2])
    drift = sim.drift_angle_init
#    lamb_ = lambratio / drift
    lamb_ = - 1/np.tan(drift)
#    lamb_ = lamb[i]
    r = r0 * np.exp(lamb_ * (pre - pre[0]))
#    r = r0 * np.exp(lamb_ * (pre - pre[0]))
    x = r*np.cos(pre)
    y = r*np.sin(pre)
    z = np.sqrt(1 - r*r)
    
#    ax.plot3D(sim.k[:,0], sim.k[:,1], sim.k[:,2])
#    ax.plot3D(x, y, z, 'k', linestyle= 'dashed')
    
    
#    plt.plot(sim.k[init:,0], sim.k[init:,1])
    plt.plot(sim.k[:,0], sim.k[:,1])
    plt.plot(x, y, 'k', linestyle='dashed')
#    leg.append(int(sim.drift_angle[0]*TODEG))

#plt.legend(leg)
#plt.plot(circle_2[0], circle_2[1], 'gray', linestyle='dashed')
#lim = [-1.1, 1.1]
lim = np.array([-1, 1])*0.75
plt.xlim(lim)
plt.ylim(lim)
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
#plt.zlabel(r'$k_z$')
plt.title(r'$(k_x, k_y)$')
plt.legend(['data', 'fit'])



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




