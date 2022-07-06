#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:19:18 2022

@author: evbernardes
"""

param_phys = {
        # masses
        'mb': 210.018 / 1000.,
        'ms': 200.685 / 1000.,
        'mp': 006.719 / 1000.,
        
        # distances
        'hp': 083./1000.,
        'hs': 145./1000.,
        
        # inertias
        'g': 9.81,
        
        'Jbx': 1.0026 / 1000.,
        'Jby': 1.0026 / 1000.,
        'Jbz': 0.1831 / 1000.,
        
        'Jsx': 0.1794 / 1000.,
        'Jsy': 0.0387 / 1000.,
        'Jsz': 0.1928 / 1000.,
        
        'Jpx': 0.0000 / 1000.,
        'Jpy': 0.0105 / 1000.,
        'Jpz': 0.1061 / 1000.,
        
        'Jringx': 2.4900 / 1000.,
        'Jringy': 2.3900 / 1000.,
        'Jringz': 4.9700 / 1000.,
        
        # motor constants
        'kf': 5.24e-06,
        'ktau': 1.08e-08,
        
        # Drag parameters
        'angvelz_ratio': 0.02, # ratio between rotor mean velocity and terminal body angular velocity
        #angvelz_ratio = 0.02 
        'angvelxy_coef': 15,
        
        'drag_linear': 0}

param_time = {
        'Nmin': 1000,
        'Nmax': None,
        'DT': 20e-4,
        'tmax': 5,
        'progress_warning_percentage': 5}

param_ctrl = {
        'alpha_lim_deg': [0, 30],
        'rotvel_ratio': 1.0,
        'rotvel_diff_lim': [0.8, 1.2],
        'P': [1.0,1.0,1.0],
        'Kp': 0.4,
        'Kd': 0.0,
        'Ki': 0.0,
        'window': 1,
        'order': 1}

param_noise = {
        'drift_angle_init_deg': 0,
        'drift_rate_rads': 0,
        'drift_flip': True,
        'noise_measures': [0, 0],
        'noise_random': [0, 0],
        'noise_impulses': [0, 0],
        'number_impulses': 0}

param_init = {
        'spin_percentage': 0.3,
        'init_orientation_zyz_deg': [0, 90, 0],
        'init_position': [0, 0, 0],
        }

param_goal = {
        'alpha_deg': 0,
        'beta_deg': 0}

#%%
