#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:20:49 2021

@author: evbernardes
"""
#%%
import os
import time
import numpy as np
import json
import quaternionic as quat
#from numpy.linalg import norm
#from scipy import integrate
import matplotlib.pyplot as plt

from monospinner_parameters import param_ctrl, param_goal, param_init, param_noise, param_phys, param_time
from monospinner_helpers import set_ctrl, set_goal, set_init, set_noise, set_phys, set_time
from helpers import wrap, cross, ad, get_middle_vector, mstack
from helpers import TODEG, TORAD, Id, zeroM, zeroV, ez, Ez, eps
from helpers_plot import plot_x_eta, plot_k_omega, plot_F


#list1 = ['param_phys', 'param_time', 'param_ctrl', 'param_noise', 'param_init', 'param_goal', 'N', 'i']

list2 = ['t', 'alpha', 'beta', 'drift_angle', 'rotvel', 'pre', 'nut', 'spin', 'alpha_lim', 'fz', 'tz']

list3 = ['k', 'k_measured', 'w', 'w_measured', 'w_delta', 
         'w_delta_int', 'w_delta_der', 't_ctrl', 'f_ctrl', 
         't_grav', 'f_grav', 't_prop', 'f_prop', 't_aero', 't_rndn', 'f_rndn', 
         't_imp', 'f_imp', 'Xd', 'angles', 'pos', 'v', 'nd_', 'nd', 'kd']

def save_param_dict(sim, datadict, param_dict_name):
    param_dict = getattr(sim, param_dict_name)
    for item in param_dict.items():
        datadict[f'{param_dict_name}_{item[0]}'] = item[1]
 
def load_param_dict(datadict, param_dict_name):
    param_dict = {}
    L = len(param_dict_name)
    for item in datadict.items():
        if(item[0][:L] == param_dict_name):
            param_dict[item[0][L+1:]] = item[1]
    return param_dict

def set_blender_file(sim, frame_step = 1, show_pos = False, frames_max = 1000):
    
    step = int(sim.N / frames_max)
    
    data = {}
    data['t'] = list(sim.t[::step])
    data['dt'] = sim.DT
    data['tmax'] = sim.tmax
    data['x'] = list(sim.pos.T[0])
    data['y'] = list(sim.pos.T[1])
    data['z'] = list(sim.pos.T[2])
    data['show_pos'] = int(show_pos)
    data['qr'] = list(sim.q.real[::step])
    data['qx'] = list(sim.q.x[::step])
    data['qy'] = list(sim.q.y[::step])
    data['qz'] = list(sim.q.z[::step])
    data['gamma'] = list(sim.angles.T[2][::step])
    #data['gamma'] = list(0*angles.T[2][::step])
    data['alpha'] = list(sim.angles.T[1][::step])
    data['beta'] = list(sim.angles.T[0][::step])
    data['frame_step'] = frame_step
    
    with open('.blender_temp.json', 'w') as outfile:
        json.dump(data, outfile)
    
#def play(self, frame_step = 1, show_pos = False, frames_max = 1000):
#    set_blender_file(self, frame_step = frame_step, show_pos = show_pos, frames_max = frames_max)
#    os.system('blender --python blender_test.py')

class Monospinner:
    def play(self, frame_step = 1, show_pos = False, frames_max = 1000):
        set_blender_file(self, frame_step = frame_step, show_pos = show_pos, frames_max = frames_max)
        os.system('blender --python blender_test.py')
        os.remove('.blender_temp.json')
    
    def __init__(self,
                 param_phys = param_phys,
                 param_time = param_time,
                 param_ctrl = param_ctrl, 
                 param_noise = param_noise, 
                 param_init = param_init, 
                 param_goal = param_goal):
        
#        %% set all elements
        set_phys(self, param_phys, False)
        set_time(self, param_time, False)
        set_ctrl(self, param_ctrl, False)
        set_noise(self, param_noise, False)
        set_init(self, param_init, False)
        set_goal(self, param_goal, False)
        
        self.init_arrays()
        

        
        
        
    def save(self, filename):
        data = {}
#        save_param_dict(self, data, 'param_phys')
#        save_param_dict(self, data, 'param_time')
#        save_param_dict(self, data, 'param_ctrl')
#        save_param_dict(self, data, 'param_noise')
#        save_param_dict(self, data, 'param_init')
#        save_param_dict(self, data, 'param_goal')
        data['param_phys'] = self.param_phys
        data['param_time'] = self.param_time
        data['param_ctrl'] = self.param_ctrl
        data['param_noise'] = self.param_noise
        data['param_init'] = self.param_init
        data['param_goal'] = self.param_goal
        data['N'] = self.N
        data['i'] = self.i
        
        for att in list2+list3+['q', 'q_measured']:
            data[f'{att}'] = getattr(self, att).tolist()
        
#        data['q'] = self.q.tolist()
#        data['q_measured'] = self.q_measured.tolist()
#        data['qr'] = list(self.q.real)
#        data['qx'] = list(self.q.x)
#        data['qy'] = list(self.q.y)
#        data['qz'] = list(self.q.z)
#        data['qmr'] = list(self.q_measured.real)
#        data['qmx'] = list(self.q_measured.x)
#        data['qmy'] = list(self.q_measured.y)
#        data['qmz'] = list(self.q_measured.z)
        
#        for att in list1:
#            data[f'{att}'] = getattr(self, att)
        
#        for att in list2:
#            data[f'{att}'] = list(getattr(self, att))
##        
#        for att in list3:
#            x, y, z = getattr(self, att).T
#            data[f'{att}_x'] = list(x)
#            data[f'{att}_y'] = list(y)
#            data[f'{att}_z'] = list(z)
            
        dirs = os.path.dirname(filename)
        
        if dirs != '' and not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)
        
#        try:
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
#            return True
#        except:
#            return data
            
        
            
    @classmethod
    def load(cls, filename):
        
        with open(filename) as json_file:
            data = json.load(json_file)
#        return data
            
        sim = cls(data['param_phys'], data['param_time'], data['param_ctrl'], 
                 data['param_noise'], data['param_init'], data['param_goal'])
        
        for att in list2+list3:
            setattr(sim, att, np.array(data[f'{att}']))
            
        for att in ['q', 'q_measured']:
            setattr(sim, att, quat.array(data[f'{att}']))
#            data[f'{att}'] = getattr(self, att).tolist()
            
#        qr = data['qr']
#        qx = data['qx']
#        qy = data['qy']
#        qz = data['qz']
#        sim.q = quat.array(np.vstack([qr, qx, qy, qz]).T)
#            
#        qmr = data['qmr']
#        qmx = data['qmx']
#        qmy = data['qmy']
#        qmz = data['qmz']
#        sim.q_measured = quat.array(np.vstack([qmr, qmx, qmy, qmz]).T)
#        
##        for att in list1:
##            setattr(sim, att, data[f'{att}'])
#        
#        sim.i = data['i']
#        sim.N = data['N']
#        
#        for att in list2:
#            setattr(sim, att, np.array(data[f'{att}']))
##            data[f'{att}'] = list(getattr(self, att))
#        
#        for att in list3:
#            x = np.array(data[f'{att}_x'])
#            y = np.array(data[f'{att}_y'])
#            z = np.array(data[f'{att}_z'])
#            setattr(sim, att, np.vstack([x,y,z]).T)
        
        return sim
        
    def init_arrays(self):
        
        N = self.N

        #%% setting impulses
        if self.number_impulses != 0:
            delta_imp = int(self.N / self.number_impulses)
            self.impulses_idx = np.array(range(self.number_impulses))*delta_imp
            self.t_imp[self.impulses_idx] = self.noise_impulses[0] * np.vstack([2*np.random.rand(2,self.number_impulses)-1,np.zeros(self.number_impulses)]).T
            self.f_imp[self.impulses_idx] = self.noise_impulses[1] * (2*np.random.rand(3,self.number_impulses).T - 1)
            
        self.t_rndn = self.noise_random[0] * np.vstack([2*np.random.rand(2,N)-1,np.zeros(N)]).T
        self.f_rndn = self.noise_random[1] * (2*np.random.rand(N,3)-1)
        
        self.i = 0
        
        #%% setting goal arrays
        self.nd = np.zeros([N,3])
        idx = np.array_split(range(N), self.ngoal)
        for i in range(self.ngoal):
            self.nd[idx[i]] = self.nd_[i]
            
        kd = (self.nd + ez).T/np.linalg.norm(self.nd + ez, axis=1)
        self.kd = kd = kd.T
        
#        #%% Forces
#        self.FZ = self.mT * self.g
#        self.fz = self.FZ * ez
#        self.tz = self.B @ self.fz
#        self.gamma_d = 1.0 * np.sqrt(self.FZ / self.kf) # rotor mean velocity

        
        #%% set initial values
        self.q[0] = self.q0
        self.drift_angle[0] = self.drift_angle_init
        self.q_drift[0] = quat.array.from_axis_angle(self.drift_angle[0] * ez)
        self.q_measured[0] = self.q[0] * self.q_drift[0]
        self.k_measured[0] = self.k[0] = get_middle_vector(self.q0)
        self.terminal_wz = self.angvelz_ratio * self.gamma_d
        self.w[0][-1] = self.init_spin_percentage * self.terminal_wz
        self.w_measured[0] = self.w[0]
        self.pos[0] = self.pos0
        self.pre[0],self.nut[0],self.spin[0] = self.q[0].to_euler_angles.T
        self.spin[0] += self.pre[0]
        
        self.rotvel[0] = self.gamma_d
        
    #%% main loop
    def run(self, stop_at_goal = None):
        
        if stop_at_goal is not None:
            stop_at_goal = np.cos(stop_at_goal*TORAD / 2)
        
        DT = self.DT
        N = self.N
        q = self.q
        k = self.k
        w = self.w
        rotvel = self.rotvel
        angles = self.angles

        pos = self.pos
        v = self.v

        
        time_start = time.time()
        time_end = time.time()
        print(f'progress: {0}%, {0}/{self.N}, elapsed: {time_end-time_start:.2f}')
        for i in range(self.i+1, N):
            
            #%% calculate all forces
            self.control(i) # attitude control
            self.motor_model(i) # get motor commands for given control
            self.gravity_and_aero(i)
            
            #%% integration
            etad = self.integrate(i)
            
            #%% reconstruction
            w[i] = w[i-1] + DT*etad[:3]
            v[i] = v[i-1] + DT*etad[3:]
            angles[i][2] = angles[i-1][2] + rotvel[i-1]*DT
            
            qd = DT * 0.5 * q[i-1] * quat.array.from_vector_part(w[i-1])
            q[i] = (q[i-1] + qd).normalized
            pos[i] = pos[i-1] + DT * q[i-1].rotate(v[i-1])
            
            k[i] = get_middle_vector(q[i], k[i-1])
            
            self.pre[i],self.nut[i],self.spin[i] = self.q[i].to_euler_angles.T
            self.spin[i] += self.pre[i]
            
            #%% noisy measurements unreliable measurements
            self.get_measured_data(i)
            
#            alpha_now = q[i].to_euler_angles[1]
            if stop_at_goal is not None and k[i].dot(self.kd[i]) >= stop_at_goal:
                break
                
                
#            if stop_at_goal and abs(self.nut[i]) < 1*TORAD:
#                break
             
            if self.gamma_d != 0 and np.linalg.norm(w[i]) > 10*self.gamma_d:
                break
            
            if i % self.dN == 0:
                alpha_now = q[i-1].to_euler_angles[1]
                time_end = time.time()
                elapsed = time_end-time_start
                elapsed_min = int(elapsed / 60)
                elapsed_sec = elapsed - 60*elapsed_min
                remaining = elapsed * (N/i - 1)
                remaining_min = int(remaining / 60)
                remaining_sec = remaining - 60*remaining_min
        #        print(f'progress: {100*i/N:.2f}%, {i}/{N}, elapsed: {elapsed:.2f}s, remaining: {remaining:.2f}s')
                print(f'progress: {100*i/N:.0f}%, '
                      f'{i}/{N}, elapsed: {elapsed_min}:{elapsed_sec:.0f}, '
                      f'remaining: {remaining_min}:{remaining_sec:.0f}, '
                      f'nut: {alpha_now*TODEG:.2f}')
        #%%
        print('****************************************')
        print('* End of test at i = {}/{}'.format(i, N-1))
        print('* Time t = {}/{}'.format(self.t[i], self.t[-1]))
        print('****************************************')
        self.i = i
        
        
        if i < N-1:
            self.trim_results(i)
        
    #%%
    def control(self, i):
        q_ = self.q_measured[i-1]
        k_ = self.k_measured[i-1]
        w_ = self.w_measured[i-1]
        kd_ = self.kd[i-1]
        n = self.ctrl_order
        
        C = np.cross(k_, kd_)
        D1 =  np.dot(k_, kd_) 
        D2 =  np.dot(k_, ez)
        
        D1 = D1 if abs(D1) > eps else 0
        D2 = D2 if abs(D2) > eps else 0
        
        S = np.sign(D1) ** (n - 1)
        
        w_spin = (k_.dot(w_)/D2)*ez
        w_ctrl = q_.inverse.rotate(C/(D1**n))
        # self.w_delta[i] = (w_.dot(w_))*ez + self.P @ q_.inverse.rotate(S * C/(D1**n))
        self.w_delta[i] = w_spin + S * self.P @ w_ctrl
        
        # angular rate control
        self.w_delta[i] = self.w_delta[i] - self.w_measured[i-1]
        self.w_delta_der[i] = (self.w_delta[i] - self.w_delta[i-1])/self.DT
        self.w_delta_int[i] = self.w_delta_int[i-1] + self.w_delta[i]*self.DT
        
#        self.t_ctrl[i] = self.Kp * self.w_delta[i-1] + self.Kd * self.w_delta_der[i-1] + self.Ki * self.w_delta_int[i-1]
        self.t_ctrl[i] = self.Kp * self.w_delta[i] + self.Kd * self.w_delta_der[i] + self.Ki * self.w_delta_int[i]
            
        self.f_ctrl[i] = -np.cross(ez, self.t_ctrl[i])/self.hp
        
        # low pass filter
        if self.ctrl_window > 1:
            self.f_ctrl[i] = np.mean(self.f_ctrl[max(0,i+1-self.ctrl_window):i+1], axis=0)
            self.t_ctrl[i] = np.mean(self.t_ctrl[max(0,i+1-self.ctrl_window):i+1], axis=0)
            
    def motor_model(self, i):
        ftot = self.f_ctrl[i] + self.fz
        
        # propeller force and torque   
        self.rotvel[i] = np.sqrt(np.linalg.norm(ftot)/self.kf)
        self.rotvel[i] = max(self.rotvel[i], self.rotvel_diff_lim[0]*self.gamma_d)
        self.rotvel[i] = min(self.rotvel[i], self.rotvel_diff_lim[1]*self.gamma_d)
        
        # get rotor angles
        ftot = ftot/np.linalg.norm(ftot)
        self.angles[i][1] = np.arccos(ftot[2])
        self.angles[i][0] = np.arctan2(ftot[1], ftot[0])
        
        if np.isnan(self.angles[i][1]):
            self.angles[i][1] = 0
        if np.isnan(self.angles[i][0]):
            self.angles[i][0] = 0
        
        # limit rotor angles
        if self.angles[i][1] < self.alpha_lim[0]:
            self.angles[i][1] = 0
        elif self.angles[i][1] > self.alpha_lim[1]:
            self.angles[i][1] = self.alpha_lim[1]
        
        Omega = 0*self.w[i-1][-1] + self.rotvel[i]
        fp = self.kf * Omega * Omega
        r = quat.array.from_euler_angles(*self.angles[i-1]).rotate(ez)
        self.f_prop[i] = fp * r
        self.t_prop[i] = self.B @ self.f_prop[i]
        
    def gravity_and_aero(self, i):
        g_B = self.q[i-1].rotate(-self.g * ez)
        self.t_grav[i] = -self.md * np.cross(ez, g_B)
        self.f_grav[i] = +self.mT * g_B
        
        # aerodynamic drag torque
        self.t_aero[i] = -np.linalg.norm(self.w[i-1]) * self.Kaero @ self.w[i-1]
#        self.t_aero[i] = -np.linalg.norm(self.w) * self.Kaero @ self.w[i-1]
        
    def integrate(self, i, gymbal_fixed = False):
        w_ = self.w[i-1]
        v_ = self.v[i-1]
        
        # all external forces and torques
        Fext = np.append(
                self.t_prop[i] + self.t_grav[i] + self.t_aero[i] + self.t_rndn[i] + self.t_imp[i],
                self.f_prop[i] + self.f_grav[i] + self.f_rndn[i] + self.f_imp[i])
        
        
        # coupling torque term
        RBP = quat.array.from_euler_angles(self.angles[i-1]).to_rotation_matrix
        t_coup = self.Jp[2,2] * np.cross(self.w[i-1], RBP @ ez)
        
        # full transformed inertia matrix
        JT = self.Jb + RBP @ self.Jp @ (RBP.T)
        
        # EOM components
        D = mstack([[JT, zeroM],[zeroM, self.mI]]) + self.mx
        Dinv = np.linalg.inv(D)
        C = -ad(w_, v_).T @ D
        
        eta = np.append(w_, v_)
        etad = Dinv @ (Fext - np.hstack([t_coup, zeroV]) - C @ eta)
        
        if gymbal_fixed:
            etad[3:] = [0, 0, 0]
        
        return etad
    
    def get_measured_data(self, i):
        self.drift_angle[i] = self.drift_angle[i-1] + self.DT*self.drift_rate
        self.q_drift[i] = quat.array.from_axis_angle(self.drift_angle[i] * ez)
        self.w_measured[i] = self.w[i] + (2*np.random.rand(3)-1)*self.noise_measures[0]
        self.q_measured[i] = self.q[i] + quat.array.random(normalize=1)*self.noise_measures[1]
        self.q_measured[i] = (self.q_measured[i]*self.q_drift[i]).normalized
        self.k_measured[i] = get_middle_vector(self.q_measured[i], self.k_measured[i])
        
    def trim_results(self, i = None):
        if i is None:
            i = self.i
        for att in list2 + list3:
            setattr(self, att, getattr(self, att)[:i])
        self.q = self.q[:i]
        self.q_measured = self.q_measured[:i]

