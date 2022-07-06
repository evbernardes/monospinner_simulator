#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:55:10 2021

@author: evbernardes
"""
import numpy as np
import json
import os
#import matplotlib.pyplot as plt
from helpers import mstack, cross
from helpers import TODEG, TORAD, Id, zeroM, zeroV, ez, Ez, eps
import quaternionic as quat

#def set_blender_file(sim, frame_step = 1, show_pos = False, frames_max = 1000):
#    
#    step = int(sim.N / frames_max)
#    
#    data = {}
#    data['t'] = list(sim.t[::step])
#    data['dt'] = sim.DT
#    data['tmax'] = sim.tmax
#    data['x'] = list(sim.pos.T[0])
#    data['y'] = list(sim.pos.T[1])
#    data['z'] = list(sim.pos.T[2])
#    data['show_pos'] = int(show_pos)
#    data['qr'] = list(sim.q.real[::step])
#    data['qx'] = list(sim.q.x[::step])
#    data['qy'] = list(sim.q.y[::step])
#    data['qz'] = list(sim.q.z[::step])
#    data['gamma'] = list(sim.angles.T[2][::step])
#    #data['gamma'] = list(0*angles.T[2][::step])
#    data['alpha'] = list(sim.angles.T[1][::step])
#    data['beta'] = list(sim.angles.T[0][::step])
#    data['frame_step'] = frame_step
#    
#    with open('sim.json', 'w') as outfile:
#        json.dump(data, outfile)
#    
#def play(frame_step = 1, show_pos = False, frames_max = 1000):
#    set_blender_file(frame_step = frame_step, show_pos = show_pos, frames_max = frames_max)
#    os.system('blender --python blender_test.py')

def get_middle_vector(q, k_old = None, e = np.array([0, 0, 1]), eps = eps):
    k = q.rotate(e) + e
    N = np.linalg.norm(k)
    
    if N < eps:
        if k_old is None:
            return np.array([0, 1, 0])
        else:
            k = k_old - (k_old.dot(e))*e
            return k / np.linalg.norm(k)
    else:    
        return k / N

#%% parameter setting helpers
def set_phys(sim, param_phys, init_arrays = True):
    sim.param_phys = param_phys
    
    g = sim.g = param_phys['g']
    hs = sim.hs = param_phys['hs']
    hp = sim.hp = param_phys['hp']
    
    mb = sim.mb = param_phys['mb']
    ms = sim.ms = param_phys['ms']
    mp = sim.mp = param_phys['mp']
    mT = sim.mT = mb+ms+mp
    md = sim.md = hs*ms - hp*mp
    sim.mI = mT * np.eye(3)
    
    sim.mx = md * mstack([[zeroM,-Ez],
                          [Ez,zeroM]])
    
    sim.r = np.array([0, 0, hp])
    sim.r.shape = (3, 1)
    sim.mpr = mp*sim.r
    sim.mprx = cross(sim.mpr.T[0])
    
    Jpx = sim.Jpx = param_phys['Jpx']
    Jpy = sim.Jpy = param_phys['Jpy']
    Jpz = sim.Jpz = param_phys['Jpz']
    Jpxy = sim.Jpxy = (Jpx+Jpy)/2
    Jp = sim.Jp = np.diag([Jpxy,Jpxy,Jpz])
    
    Jbx = sim.Jbx = param_phys['Jbx']
    Jby = sim.Jby = param_phys['Jby']
    Jbz = sim.Jbz = param_phys['Jbz']
    Jb = sim.Jb = np.diag([Jbx,Jby,Jbz])
    
    Jsx = sim.Jsx = param_phys['Jsx']
    Jsy = sim.Jsy = param_phys['Jsy']
    Jsz = sim.Jsz = param_phys['Jsz']
    Js = sim.Js = np.diag([Jsx,Jsy,Jsz])
    
    sim.Jb = sim.Jb + Js + np.diag([1,1,0])*(ms*(hs**2) + mp*(hp**2))
    
    sim.ktau = param_phys['ktau']
    sim.kf = param_phys['kf']
    sim.kratio = sim.ktau/sim.kf
    
    sim.B = sim.kratio*Id + hp*Ez
    sim.Binv = np.linalg.inv(sim.B)
    sim.BTotal = sim.Binv @ sim.Jb / sim.kf
    
    sim.angvelz_ratio = param_phys['angvelz_ratio']
    sim.angvelxy_coef = param_phys['angvelxy_coef']
#    sim.angvelz_ratio = param_phys['angvelz_ratio']
    
    # Drag
    if sim.angvelz_ratio == 0:
        sim.Kz = 1000000
    else:
        sim.Kz = sim.ktau * (1/sim.angvelz_ratio + 1)**2
        
    sim.Kaero = np.diag([sim.angvelxy_coef, sim.angvelxy_coef, 1]) * sim.Kz
    
    #%% Forces
    sim.FZ = sim.mT * sim.g
    sim.fz = sim.FZ * ez
    sim.tz = sim.B @ sim.fz
    sim.gamma_d = 1.0 * np.sqrt(sim.FZ / sim.kf) # rotor mean velocity
    
    if init_arrays: sim.init_arrays()

def set_time(sim, param_time, init_arrays = True):
    
    sim.param_time = param_time
    
    sim.Nmin = Nmin = param_time['Nmin']
    sim.Nmin = Nmax = param_time['Nmax']
    sim.DT = DT = param_time['DT']
    sim.tmax = tmax = param_time['tmax']

    sim.t = np.arange(0, tmax+DT, DT)
    sim.N = N = len(sim.t)
    
    if Nmin is not None and N < Nmin:
        sim.DT = tmax/Nmin
        sim.t = np.array(range(Nmin))*DT
        sim.N = Nmin
    elif Nmax is not None and N > Nmax:
        sim.DT = tmax/Nmax
        sim.t = np.array(range(Nmax))*DT
        sim.N = Nmax
    sim.dN = int(N / (100/param_time['progress_warning_percentage']))
    
    # setting zero arrays
    sim.q = quat.array(np.zeros([N,4]))
    sim.q_measured = quat.array(np.zeros([N,4]))
    sim.q_drift = quat.array(np.zeros([N,4]))
    sim.k = np.zeros([N,3])
    sim.k_measured = np.zeros([N,3])
    sim.w = np.zeros([N,3])
    sim.w_measured = np.zeros([N,3])
    sim.w_delta = np.zeros([N,3])
    sim.w_delta_int = np.zeros([N,3])
    sim.w_delta_der = np.zeros([N,3])
    sim.t_ctrl = np.zeros([N,3])
    sim.f_ctrl = np.zeros([N,3])
    sim.t_grav = np.zeros([N,3])
    sim.f_grav = np.zeros([N,3])
    sim.t_prop = np.zeros([N,3])
    sim.f_prop = np.zeros([N,3])
    sim.t_aero = np.zeros([N,3])
    sim.t_rndn = np.zeros([N,3])
    sim.f_rndn = np.zeros([N,3])
    sim.t_imp = np.zeros([N,3])
    sim.f_imp = np.zeros([N,3])
#    sim.Fext = np.zeros(6)
    sim.alpha = np.zeros(N)
    sim.beta = np.zeros(N)
    sim.Xd = np.zeros([N,3])
    sim.drift_angle = np.zeros(N)
    sim.angles = np.zeros([N,3])
    sim.rotvel = np.zeros(N)
    sim.pos = np.zeros([N,3])
    sim.v = np.zeros([N,3])
    sim.pre = np.zeros(N)
    sim.nut = np.zeros(N)
    sim.spin = np.zeros(N)
    
    if init_arrays: sim.init_arrays()

def set_ctrl(sim, param_ctrl, init_arrays = True):
    sim.param_ctrl = param_ctrl
    
    # set control parameters
    sim.ctrl_window = param_ctrl['window']
    sim.alpha_lim = np.array(param_ctrl['alpha_lim_deg'])*TORAD
    sim.rotvel_diff_lim = param_ctrl['rotvel_diff_lim']
    
    # control parameters
    sim.P = np.diag(param_ctrl['P']) * sim.mT * sim.g
    sim.Kp = param_ctrl['Kp'] 
    sim.Kd = param_ctrl['Kd']
    sim.Ki = param_ctrl['Ki']
    sim.ctrl_order = param_ctrl['order']
    
    if init_arrays: sim.init_arrays()
    
def set_noise(sim, param_noise, init_arrays = True):
    sim.param_noise = param_noise
    
    # set noise
#        sim.drift_is_constant = sim.param_noise['drift_is_constant']
    sim.drift_rate = param_noise['drift_rate_rads']
    sim.drift_flip = param_noise['drift_flip']
    sim.drift_angle_init = param_noise['drift_angle_init_deg']*TORAD
    
    sim.noise_measures = sim.param_noise['noise_measures']
    sim.noise_random = sim.param_noise['noise_random']
    sim.noise_impulses = sim.param_noise['noise_impulses']
    
    sim.number_impulses = sim.param_noise['number_impulses']
    
    if init_arrays: sim.init_arrays()
    
def set_init(sim, param_init, init_arrays = True):
    sim.param_init = param_init
    
    # initual condition
    zyz = np.array(sim.param_init['init_orientation_zyz_deg'])*TORAD
    sim.q0 = quat.array.from_euler_angles(*zyz)
    sim.pos0 = sim.param_init['init_position']
    sim.init_spin_percentage = 0.3
    
    if init_arrays: sim.init_arrays()
    
def set_goal(sim, param_goal, init_arrays = True):
    sim.param_goal = param_goal
    
    # goal orientation

    alphad = sim.param_goal['alpha_deg']
    betad = sim.param_goal['beta_deg']
    
    if type(alphad) != list: alphad = [alphad]
    if type(betad) != list: betad = [betad]
    
    ngoal = len(alphad)
    if ngoal != len(betad):
        raise ValueError('alphad and betad must have the same length')
    
    alphad = np.array(alphad)*TORAD
    betad = np.array(betad)*TORAD
    sim.ngoal = ngoal
    
    sa, ca = np.sin(alphad), np.cos(alphad)
    sb, cb = np.sin(betad), np.cos(betad)
    
    sim.nd_ = np.array([sa*cb, sa*sb, ca]).T
    
    if init_arrays: sim.init_arrays()
    
#def trim_results(sim, i = None):
#    if i is None:
#        i = sim.i
#    sim.t = sim.t[:i]
#    sim.q = sim.q[:i]
#    sim.q_measured = sim.q_measured[:i]
#    sim.q_drift = sim.q_drift[:i]
#    sim.k = sim.k[:i]
#    sim.kd = sim.kd[:i]
#    sim.k_measured = sim.k_measured[:i]
#    sim.w = sim.w[:i]
#    sim.w_measured = sim.w_measured[:i]
#    sim.w_delta = sim.w_delta[:i]
#    sim.w_delta_int = sim.w_delta_int[:i]
#    sim.w_delta_der = sim.w_delta_der[:i]
#    sim.t_ctrl = sim.t_ctrl[:i]
#    sim.f_ctrl = sim.f_ctrl[:i]
#    sim.t_grav = sim.t_grav[:i]
#    sim.f_grav = sim.f_grav[:i]
#    sim.t_prop = sim.t_prop[:i]
#    sim.f_prop = sim.f_prop[:i]
#    sim.t_aero = sim.t_aero[:i]
#    sim.t_rndn = sim.t_rndn[:i]
#    sim.f_rndn = sim.f_rndn[:i]
#    sim.t_imp = sim.t_imp[:i]
#    sim.f_imp = sim.f_imp[:i]
#    
#    sim.alpha = sim.alpha[:i]
#    sim.beta = sim.beta[:i]
#    sim.Xd = sim.Xd[:i]
#    sim.drift_angle = sim.drift_angle[:i]
#    sim.angles = sim.angles[:i]
#    sim.rotvel = sim.rotvel[:i]
#    sim.pos = sim.pos[:i]
#    sim.v = sim.v[:i]
#    sim.pre = sim.pre[:i]
#    sim.nut = sim.nut[:i]
#    sim.spin = sim.spin[:i]
#    sim.N = len(sim.t)
